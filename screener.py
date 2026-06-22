"""
screener.py
===========
スクリーニング実行モジュール。

学習済みモデルと直近データを使って「今、急騰する確率が高い銘柄」を選定する。

出力フォーマット:
- コンソール: 上位N銘柄をテーブル表示
- CSV: 全銘柄のスコアをスタンプ付きで保存

設計方針:
- モデルとスケーラーはjoblibで読み込む（学習不要で即座にスクリーニング可能）
- 最新の特徴量は直近LOOKBACK_DAYS日分のデータから計算
- スコアリングは全銘柄の直近1行（最新日）を対象
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_fetcher import get_latest_data, fetch_fundamental_data
from features import compute_features, get_feature_columns, build_feature_dataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  スクリーニング実行
# ------------------------------------------------------------------ #

def load_model_and_scaler(model_path: str, scaler_path: str):
    """保存済みモデルとスケーラーを読み込む。"""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run main.py with mode='train' first."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded model from {model_path}")
    return model, scaler


def extract_latest_features(
    dataset: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    各銘柄の最新日の特徴量を抽出する。

    Parameters
    ----------
    dataset : pd.DataFrame
        インデックス=(ticker, date)のMultiIndex DataFrame
    feature_cols : list[str]
        使用する特徴量カラム

    Returns
    -------
    pd.DataFrame
        インデックス=ticker、カラム=特徴量（最新日のスナップショット）
    """
    available_cols = [c for c in feature_cols if c in dataset.columns]
    latest_rows = []

    for ticker in dataset.index.get_level_values("ticker").unique():
        try:
            ticker_data = dataset.loc[ticker, available_cols]
            if ticker_data.empty:
                continue
            latest = ticker_data.iloc[-1]  # 最新日の行
            latest.name = ticker
            latest_rows.append(latest)
        except Exception as e:
            logger.warning(f"Failed to extract features for {ticker}: {e}")

    if not latest_rows:
        return pd.DataFrame()

    df_latest = pd.DataFrame(latest_rows)
    df_latest.index.name = "ticker"
    return df_latest


def run_screener(
    tickers: list[str],
    model_path: str = "./models/model.pkl",
    scaler_path: str = "./models/scaler.pkl",
    feature_params: Optional[dict] = None,
    lookback_days: int = 120,
    top_n: int = 20,
    output_dir: str = "./results",
    plot: bool = True,
) -> pd.DataFrame:
    """
    スクリーニングを実行し、急騰確率の高い銘柄リストを返す。

    Parameters
    ----------
    tickers : list[str]
        スクリーニング対象のティッカーリスト
    model_path : str
        学習済みモデルのパス
    scaler_path : str
        スケーラーのパス
    feature_params : dict, optional
        特徴量計算パラメータ
    lookback_days : int
        直近何日分のデータを取得するか（指標計算のウォームアップ期間を含む）
    top_n : int
        表示する上位銘柄数
    output_dir : str
        結果CSV・グラフの保存先
    plot : bool
        Trueの場合、スコア分布グラフを表示・保存

    Returns
    -------
    pd.DataFrame
        銘柄ごとの急騰確率スコアと主要シグナル指標
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. モデルの読み込み
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # 2. 直近データの取得
    logger.info(f"Fetching latest data for {len(tickers)} tickers...")
    latest_price_data = get_latest_data(tickers, lookback_days=lookback_days)

    if not latest_price_data:
        logger.error("No price data fetched for screening.")
        return pd.DataFrame()

    # 3. 特徴量の計算
    logger.info("Computing features...")
    dataset = build_feature_dataset(latest_price_data, params=feature_params)

    if dataset.empty:
        logger.error("Feature dataset is empty.")
        return pd.DataFrame()

    # 4. 各銘柄の最新特徴量を抽出
    feature_cols = get_feature_columns()
    X_latest = extract_latest_features(dataset, feature_cols)

    if X_latest.empty:
        logger.error("No latest features extracted.")
        return pd.DataFrame()

    # 5. スケーリングと予測
    available_cols = [c for c in feature_cols if c in X_latest.columns]
    X_for_pred = X_latest[available_cols].fillna(0)

    # 訓練時のカラム順序に合わせる（モデルがカラム名を記憶している場合）
    try:
        X_scaled = pd.DataFrame(
            scaler.transform(X_for_pred),
            index=X_for_pred.index,
            columns=X_for_pred.columns,
        )
    except Exception as e:
        logger.warning(f"Scaler transform failed: {e}. Using raw features.")
        X_scaled = X_for_pred

    # 急騰確率（正例の予測確率）
    breakout_proba = model.predict_proba(X_scaled)[:, 1]

    # 6. 結果DataFrame の構築
    result_df = pd.DataFrame({"breakout_probability": breakout_proba}, index=X_latest.index)

    # 主要シグナル指標を追加（スクリーニング結果の解釈補助）
    signal_cols = [
        "rsi", "macd_hist", "bb_squeeze_ratio", "volume_ratio",
        "breakout_composite", "squeeze_expansion",
        "ma_short_dev", "ma_mid_dev",
    ]
    for col in signal_cols:
        if col in X_latest.columns:
            result_df[col] = X_latest[col]

    # 直近の終値・出来高も参考情報として追加
    close_map = {}
    volume_map = {}
    for ticker, df in latest_price_data.items():
        if not df.empty:
            close_map[ticker] = df["Close"].iloc[-1]
            volume_map[ticker] = df["Volume"].iloc[-1]

    result_df["latest_close"] = pd.Series(close_map)
    result_df["latest_volume"] = pd.Series(volume_map)

    # スコア降順でソート
    result_df = result_df.sort_values("breakout_probability", ascending=False)

    # 7. 出力
    _display_top_results(result_df, top_n)
    _save_results(result_df, output_dir)

    if plot:
        _plot_screening_results(result_df, top_n, output_dir)

    return result_df


def _display_top_results(result_df: pd.DataFrame, top_n: int) -> None:
    """スクリーニング結果をコンソールに表示する。"""
    print("\n" + "=" * 70)
    print(f"  BREAKOUT SCREENING RESULTS - Top {top_n} Candidates")
    print(f"  Screened at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    display_cols = ["breakout_probability", "latest_close", "rsi",
                    "volume_ratio", "bb_squeeze_ratio", "breakout_composite"]
    available = [c for c in display_cols if c in result_df.columns]

    top = result_df.head(top_n)[available].copy()
    top["breakout_probability"] = top["breakout_probability"].map("{:.1%}".format)
    if "latest_close" in top.columns:
        top["latest_close"] = top["latest_close"].map("{:,.0f}".format)
    if "rsi" in top.columns:
        top["rsi"] = top["rsi"].map("{:.1f}".format)
    if "volume_ratio" in top.columns:
        top["volume_ratio"] = top["volume_ratio"].map("{:.2f}x".format)
    if "bb_squeeze_ratio" in top.columns:
        top["bb_squeeze_ratio"] = top["bb_squeeze_ratio"].map("{:.2f}".format)

    print(top.to_string())
    print("=" * 70)

    # アラートレベル別に銘柄を分類して表示
    high_prob = result_df[result_df["breakout_probability"] >= 0.7]
    mid_prob = result_df[(result_df["breakout_probability"] >= 0.5) &
                          (result_df["breakout_probability"] < 0.7)]

    print(f"\n  HIGH ALERT (>= 70%): {len(high_prob)} tickers")
    if not high_prob.empty:
        print(f"    {', '.join(high_prob.index.tolist()[:10])}")

    print(f"  WATCH (50-70%): {len(mid_prob)} tickers")
    if not mid_prob.empty:
        print(f"    {', '.join(mid_prob.index.tolist()[:10])}")
    print()


def _save_results(result_df: pd.DataFrame, output_dir: str) -> None:
    """スクリーニング結果をCSVに保存する。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"screening_result_{timestamp}.csv"
    result_df.to_csv(csv_path)
    logger.info(f"Results saved to {csv_path}")

    # 最新結果は固定ファイル名でも保存（最新参照用）
    latest_path = Path(output_dir) / "screening_result_latest.csv"
    result_df.to_csv(latest_path)


def _plot_screening_results(
    result_df: pd.DataFrame,
    top_n: int,
    output_dir: str,
) -> None:
    """スクリーニング結果のグラフを作成・保存する。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 上位N銘柄の急騰確率バーチャート
    top = result_df.head(top_n)
    colors = ["#e74c3c" if p >= 0.7 else "#e67e22" if p >= 0.5 else "#3498db"
              for p in top["breakout_probability"]]
    axes[0].barh(top.index[::-1], top["breakout_probability"][::-1], color=colors[::-1])
    axes[0].axvline(0.5, color="orange", linestyle="--", alpha=0.7, label="50% threshold")
    axes[0].axvline(0.7, color="red", linestyle="--", alpha=0.7, label="70% threshold")
    axes[0].set_xlabel("Breakout Probability")
    axes[0].set_title(f"Top {top_n} Breakout Candidates")
    axes[0].legend()

    # 右: 全銘柄のスコア分布
    axes[1].hist(result_df["breakout_probability"], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axvline(0.5, color="orange", linestyle="--", alpha=0.8, label="50%")
    axes[1].axvline(0.7, color="red", linestyle="--", alpha=0.8, label="70%")
    axes[1].set_xlabel("Breakout Probability")
    axes[1].set_ylabel("Number of Tickers")
    axes[1].set_title("Score Distribution (All Tickers)")
    axes[1].legend()

    plt.suptitle(
        f"Breakout Screening Results - {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14
    )
    plt.tight_layout()

    save_path = Path(output_dir) / "screening_chart_latest.png"
    plt.savefig(save_path, dpi=150)
    logger.info(f"Chart saved to {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  シグナル別フィルタリング（追加スクリーニング条件）
# ------------------------------------------------------------------ #

def filter_by_signals(
    result_df: pd.DataFrame,
    min_probability: float = 0.5,
    require_volume_spike: bool = True,
    require_squeeze: bool = False,
    min_rsi: float = 40.0,
    max_rsi: float = 80.0,
) -> pd.DataFrame:
    """
    MLスコアに加えてルールベースの条件でさらに絞り込む。

    MLモデルが「潜在的な急騰候補」を拾い、ルールベースが「シグナル強度」で確認する
    2段階スクリーニングとして使用可能。

    Parameters
    ----------
    min_probability : float
        最低急騰確率（デフォルト: 50%以上）
    require_volume_spike : bool
        出来高急増（volume_ratio > 2.0）を必須条件にするか
    require_squeeze : bool
        ボリンジャーバンドのスクイーズ解放を必須条件にするか
    min_rsi, max_rsi : float
        RSIの許容範囲（過買い・過売りゾーンを除外）
    """
    filtered = result_df[result_df["breakout_probability"] >= min_probability].copy()

    if "rsi" in filtered.columns:
        filtered = filtered[
            (filtered["rsi"] >= min_rsi) & (filtered["rsi"] <= max_rsi)
        ]

    if require_volume_spike and "volume_ratio" in filtered.columns:
        filtered = filtered[filtered["volume_ratio"] >= 2.0]

    if require_squeeze and "squeeze_expansion" in filtered.columns:
        filtered = filtered[filtered["squeeze_expansion"] == 1]

    logger.info(f"Filtered: {len(result_df)} → {len(filtered)} tickers")
    return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_fetcher import DEFAULT_TICKERS

    result = run_screener(
        tickers=DEFAULT_TICKERS[:10],  # テスト用に10銘柄
        top_n=10,
    )
