"""
model.py
========
機械学習モデルの学習・評価・保存モジュール。

設計方針:
- ラベル（ターゲット変数）の定義はN日後X%上昇という形式でパラメータ化
- 時系列クロスバリデーション（TimeSeriesSplit）で将来データリークを防ぐ
- LightGBMとRandomForestを切り替え可能な構造
- 学習済みモデルはjoblibで保存し、screener.pyから再利用可能
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from features import get_feature_columns

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  デフォルトパラメータ
# ------------------------------------------------------------------ #

DEFAULT_MODEL_PARAMS = {
    # ラベリング条件
    "label_horizon": 20,      # N日後（例: 20営業日 ≒ 1ヶ月）
    "label_threshold": 0.15,  # X%以上上昇したら正例（例: 15%）

    # モデル選択: "lightgbm" or "random_forest"
    "model_type": "lightgbm",

    # LightGBMパラメータ
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    },

    # RandomForestパラメータ
    "rf_params": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",  # クラス不均衡対策
        "random_state": 42,
        "n_jobs": -1,
    },

    # 時系列クロスバリデーション
    "cv_n_splits": 5,

    # モデル保存先
    "model_save_path": "./models/model.pkl",
    "scaler_save_path": "./models/scaler.pkl",
}


# ------------------------------------------------------------------ #
#  ラベリング
# ------------------------------------------------------------------ #

def create_labels(
    dataset: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.Series:
    """
    「N日後にX%以上上昇したか」の2値ラベルを生成する。

    Parameters
    ----------
    dataset : pd.DataFrame
        インデックス=(ticker, date)のMultiIndex DataFrame（Closeカラムを含む）
    horizon : int
        何営業日後のリターンで判定するか
    threshold : float
        正例とするリターン閾値（例: 0.15 = 15%）

    Returns
    -------
    pd.Series
        1（上昇）or 0（未達成）のラベル。未来データが存在しない末尾はNaN。
    """
    labels = {}

    for ticker in dataset.index.get_level_values("ticker").unique():
        try:
            ticker_df = dataset.loc[ticker, "Close"].sort_index()
        except KeyError:
            continue

        # N日後の終値 / 現在の終値 - 1 = 将来リターン
        future_return = ticker_df.shift(-horizon) / ticker_df - 1
        label = (future_return >= threshold).astype(float)
        # 将来データが存在しない末尾はNaNのまま
        label[future_return.isna()] = np.nan

        for date, val in label.items():
            labels[(ticker, date)] = val

    result = pd.Series(labels, name="label")
    result.index.names = ["ticker", "date"]

    positive_rate = result.dropna().mean()
    logger.info(
        f"Labels created: {result.notna().sum()} samples, "
        f"positive rate: {positive_rate:.1%} "
        f"(horizon={horizon}d, threshold={threshold:.0%})"
    )

    return result


# ------------------------------------------------------------------ #
#  データ準備
# ------------------------------------------------------------------ #

def prepare_train_data(
    dataset: pd.DataFrame,
    labels: pd.Series,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    特徴量とラベルを結合し、学習用データを準備する。

    - NaNラベルの行（未来データ不足）を除外
    - 特徴量のNaNをゼロ埋め（LightGBMはNaN許容だが念のため）
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    available_cols = [c for c in feature_cols if c in dataset.columns]
    X = dataset[available_cols].copy()

    # ラベルと結合
    X = X.join(labels, how="inner")
    X = X.dropna(subset=["label"])

    y = X.pop("label").astype(int)

    logger.info(f"Training data: {len(X)} samples, {len(available_cols)} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y


# ------------------------------------------------------------------ #
#  モデル学習
# ------------------------------------------------------------------ #

def build_model(model_type: str, params: dict):
    """
    モデルオブジェクトを構築して返す。
    """
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'lightgbm' or 'random_forest'.")


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    model_params: dict,
    n_splits: int = 5,
) -> dict:
    """
    TimeSeriesSplitによるクロスバリデーション。

    時系列データの性質を考慮し、常に過去データで学習→未来データで評価する分割を使用。
    これにより将来データのリーク（データリーク）を防ぐ。

    Returns
    -------
    dict
        各フォールドの評価指標と、全フォールドの平均値
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == "lightgbm":
            model_params_copy = {
                k: v for k, v in model_params.items()
                if k not in ["n_estimators"]  # early_stoppingと競合するため別途指定
            }
            model = lgb.LGBMClassifier(
                n_estimators=model_params.get("n_estimators", 500),
                **{k: v for k, v in model_params.items() if k != "n_estimators"}
            )
            # LightGBMのみearly stoppingを使用
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
            )
        else:
            model = build_model(model_type, model_params)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "fold": fold + 1,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob) if y_val.nunique() > 1 else 0.5,
        }
        fold_results.append(metrics)

        # 特徴量重要度の記録
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns, name=f"fold_{fold+1}")
            feature_importances.append(fi)

        logger.info(
            f"Fold {fold+1}: Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
        )

    # 平均スコアの計算
    results_df = pd.DataFrame(fold_results)
    mean_metrics = results_df[["precision", "recall", "f1", "roc_auc"]].mean()

    print("\n" + "=" * 50)
    print("Cross-Validation Results (Mean ± Std)")
    print("=" * 50)
    for col in ["precision", "recall", "f1", "roc_auc"]:
        mean = results_df[col].mean()
        std = results_df[col].std()
        print(f"  {col:<15}: {mean:.3f} ± {std:.3f}")
    print("=" * 50)

    # 特徴量重要度の集計
    fi_df = None
    if feature_importances:
        fi_df = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    return {
        "fold_results": fold_results,
        "mean_metrics": mean_metrics.to_dict(),
        "feature_importance": fi_df,
    }


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    model_params: dict,
    save_path: str,
    scaler_path: str,
) -> tuple:
    """
    全データで最終モデルを学習し、保存する。

    Returns
    -------
    tuple
        (学習済みモデル, StandardScaler)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # スケーリング（ランダムフォレストは不要だが統一的に処理）
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.fillna(0)),
        index=X.index,
        columns=X.columns,
    )

    model = build_model(model_type, model_params)
    model.fit(X_scaled, y)

    # 保存
    joblib.dump(model, save_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Model saved to {save_path}")
    logger.info(f"Scaler saved to {scaler_path}")

    return model, scaler


# ------------------------------------------------------------------ #
#  可視化
# ------------------------------------------------------------------ #

def plot_feature_importance(
    feature_importance: pd.Series,
    top_n: int = 20,
    save_path: Optional[str] = "./models/feature_importance.png",
) -> None:
    """特徴量重要度をバープロットで可視化する。"""
    top_fi = feature_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top_fi.values, y=top_fi.index, palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (Mean over CV Folds)", fontsize=14)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_cv_results(fold_results: list[dict], save_path: Optional[str] = "./models/cv_results.png") -> None:
    """クロスバリデーション結果をフォールドごとに可視化する。"""
    df = pd.DataFrame(fold_results)
    metrics = ["precision", "recall", "f1", "roc_auc"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        ax.bar(df["fold"], df[metric], color="steelblue", alpha=0.7)
        ax.axhline(df[metric].mean(), color="red", linestyle="--", label=f"Mean: {df[metric].mean():.3f}")
        ax.set_title(metric.upper())
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.suptitle("Time Series Cross-Validation Results", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"CV results plot saved to {save_path}")

    plt.show()


# ------------------------------------------------------------------ #
#  メインパイプライン
# ------------------------------------------------------------------ #

def run_model_pipeline(
    dataset: pd.DataFrame,
    params: Optional[dict] = None,
    plot: bool = True,
) -> tuple:
    """
    ラベリング → 学習 → 評価 → 保存の一連パイプラインを実行する。

    Parameters
    ----------
    dataset : pd.DataFrame
        build_feature_dataset()の出力（Closeカラムを含む）
    params : dict, optional
        モデルパラメータ。NoneはDEFAULT_MODEL_PARAMSを使用
    plot : bool
        Trueの場合、グラフを表示・保存する

    Returns
    -------
    tuple
        (学習済みモデル, スケーラー, CV評価結果dict)
    """
    if params is None:
        params = DEFAULT_MODEL_PARAMS

    # 1. ラベルの生成
    labels = create_labels(
        dataset,
        horizon=params["label_horizon"],
        threshold=params["label_threshold"],
    )

    # 2. 学習データの準備
    X, y = prepare_train_data(dataset, labels)

    if len(X) < 100:
        raise ValueError(f"Insufficient training samples: {len(X)}. Expand date range or add more tickers.")

    # 3. クロスバリデーション
    model_type = params["model_type"]
    model_params = params.get(f"{model_type.replace('random_forest', 'rf')}_params",
                               params.get("lgb_params"))

    cv_results = cross_validate_model(
        X, y,
        model_type=model_type,
        model_params=model_params,
        n_splits=params["cv_n_splits"],
    )

    # 4. 可視化
    if plot and cv_results["feature_importance"] is not None:
        plot_feature_importance(cv_results["feature_importance"])
        plot_cv_results(cv_results["fold_results"])

    # 5. 最終モデルの学習と保存
    model, scaler = train_final_model(
        X, y,
        model_type=model_type,
        model_params=model_params,
        save_path=params["model_save_path"],
        scaler_path=params["scaler_save_path"],
    )

    return model, scaler, cv_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("model.py: Run via main.py for full pipeline execution.")
