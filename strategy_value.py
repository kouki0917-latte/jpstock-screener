"""
strategy_value.py
=================
「格安・割安 × 1週間10%反転」戦略モジュール。

戦略の思想:
  安値圏に叩き売られた割安株が底値を打ち、
  短期的に10%以上リバウンドする銘柄を毎日スクリーニングする。
  この10%ゲインを繰り返し積み上げることで資産を複利成長させる。

既存の breakout screener（中長期20日/15%）とは独立したモデルとして並走する。

特徴量の設計:
  - 割安度: 52週安値からの距離、移動平均からの乖離（下方）
  - 売られすぎ度: RSI・Stochastics・Williams %R の複合
  - 買い圧力の萌芽: MFI（Money Flow Index）上向き転換、OBV変化
  - 出来高の変化: 底値圏での出来高増加 = 機関の拾い始めシグナル
  - 短期モメンタム反転: 1日・3日リターンが負→正に転換しかけている
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  パラメータ
# ------------------------------------------------------------------ #

DEFAULT_VALUE_PARAMS = {
    # ラベリング: 5営業日後（1週間）に10%以上上昇
    "label_horizon": 5,
    "label_threshold": 0.10,

    # 特徴量ウィンドウ
    "rsi_window": 14,
    "stoch_window": 14,
    "stoch_smooth": 3,
    "williams_window": 14,
    "mfi_window": 14,
    "obv_ma_window": 10,
    "bb_window": 20,
    "bb_std": 2.0,
    "ma_short": 5,
    "ma_mid": 25,
    "ma_long": 75,
    "volume_window": 20,
    "week_high_low_window": 52 * 5,  # 52週 ≈ 260営業日

    # モデル
    "cv_n_splits": 5,
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "scale_pos_weight": 3,  # 正例が少ないためクラス重みを強化
    },

    "model_save_path": "./models/value_model.pkl",
    "scaler_save_path": "./models/value_scaler.pkl",
    "min_rows": 130,  # 52週安値計算に必要な最低行数
}


# ------------------------------------------------------------------ #
#  特徴量計算
# ------------------------------------------------------------------ #

def calc_value_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    割安・反転特化の特徴量を計算する。

    【割安度指標】
    - dist_from_52w_low: 52週安値からの距離（低いほど割安）
    - dist_from_52w_high: 52週高値からの距離（高値からどれだけ下落したか）
    - bb_lower_dist: ボリンジャー下限からの距離（下限付近 = 売られすぎゾーン）

    【売られすぎ指標】
    - rsi: 30以下で売られすぎ
    - stoch_k, stoch_d: ストキャスティクス（20以下で売られすぎ）
    - williams_r: -80以下で売られすぎ
    - mfi: 20以下で売られすぎ（出来高加重のRSI）

    【反転シグナル】
    - stoch_cross_up: Stoch %Kが%Dを上抜け（底値圏でのゴールデンクロス）
    - mfi_turning_up: MFIが3日連続上昇（資金流入の始まり）
    - obv_ma_diff: OBV と OBV移動平均の乖離（正転 = 買い優勢）
    - reversal_candle: 下ヒゲが長い陽線（ハンマー足）

    【モメンタム状態】
    - return_1d, return_3d: 直近リターン（マイナス = まだ下落中）
    - return_1d_positive: 本日リターンが正転（反転の初日を捉える）
    """
    df = df.copy()

    # --- 割安度: 52週高値/安値からの距離 ---
    w = params["week_high_low_window"]
    rolling_low = df["Low"].rolling(window=min(w, len(df))).min()
    rolling_high = df["High"].rolling(window=min(w, len(df))).max()

    df["dist_from_52w_low"] = (df["Close"] - rolling_low) / rolling_low.replace(0, np.nan)
    df["dist_from_52w_high"] = (df["Close"] - rolling_high) / rolling_high.replace(0, np.nan)

    # 52週安値圏にいるか（下位20%以内）
    high_low_range = (rolling_high - rolling_low).replace(0, np.nan)
    df["pct_in_52w_range"] = (df["Close"] - rolling_low) / high_low_range

    # --- ボリンジャー ---
    bb = ta.volatility.BollingerBands(
        close=df["Close"], window=params["bb_window"], window_dev=params["bb_std"], fillna=False
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_pct"] = bb.bollinger_pband()  # 0=下限, 1=上限
    df["bb_lower_dist"] = (df["Close"] - df["bb_lower"]) / df["bb_lower"].replace(0, np.nan)

    # --- RSI ---
    rsi_ind = ta.momentum.RSIIndicator(close=df["Close"], window=params["rsi_window"], fillna=False)
    df["rsi"] = rsi_ind.rsi()
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    df["rsi_turning"] = (
        (df["rsi"].shift(1) < 30) & (df["rsi"] > df["rsi"].shift(1))
    ).astype(int)  # 売られすぎから上向き転換

    # --- Stochastic Oscillator ---
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=params["stoch_window"], smooth_window=params["stoch_smooth"], fillna=False
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)

    # %Kが%Dを下から上抜け（ゴールデンクロス）かつ売られすぎゾーン内
    prev_k_below = df["stoch_k"].shift(1) < df["stoch_d"].shift(1)
    curr_k_above = df["stoch_k"] >= df["stoch_d"]
    df["stoch_cross_up"] = (prev_k_below & curr_k_above & (df["stoch_k"] < 40)).astype(int)

    # --- Williams %R ---
    williams = ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"],
        lbp=params["williams_window"], fillna=False
    )
    df["williams_r"] = williams.williams_r()
    df["williams_oversold"] = (df["williams_r"] < -80).astype(int)

    # --- MFI（Money Flow Index） ---
    mfi = ta.volume.MFIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"],
        volume=df["Volume"], window=params["mfi_window"], fillna=False
    )
    df["mfi"] = mfi.money_flow_index()
    df["mfi_oversold"] = (df["mfi"] < 25).astype(int)
    df["mfi_turning_up"] = (
        (df["mfi"] > df["mfi"].shift(1)) &
        (df["mfi"].shift(1) > df["mfi"].shift(2)) &
        (df["mfi"] < 50)
    ).astype(int)  # 低位から3日連続上昇

    # --- OBV（On-Balance Volume） ---
    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"], fillna=False
    )
    df["obv"] = obv.on_balance_volume()
    df["obv_ma"] = df["obv"].rolling(window=params["obv_ma_window"]).mean()
    df["obv_ma_diff"] = df["obv"] - df["obv_ma"]  # 正 = OBVがMAを上回る = 買い優勢
    df["obv_rising"] = (df["obv"] > df["obv"].shift(3)).astype(int)

    # --- 移動平均 ---
    df["ma_short"] = df["Close"].rolling(params["ma_short"]).mean()
    df["ma_mid"] = df["Close"].rolling(params["ma_mid"]).mean()
    df["ma_long"] = df["Close"].rolling(params["ma_long"]).mean()

    # 移動平均からの乖離（負 = 下方乖離 = 割安候補）
    df["ma_short_dev"] = (df["Close"] - df["ma_short"]) / df["ma_short"].replace(0, np.nan)
    df["ma_mid_dev"] = (df["Close"] - df["ma_mid"]) / df["ma_mid"].replace(0, np.nan)
    # 短期MAが中期MAを下抜けているか（デッドクロス状態 = 下落トレンド中）
    df["in_downtrend"] = (df["ma_short"] < df["ma_mid"]).astype(int)

    # --- リターン ---
    df["return_1d"] = df["Close"].pct_change()
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_1d_positive"] = (df["return_1d"] > 0).astype(int)  # 反転初日フラグ

    # --- 出来高 ---
    df["volume_ma"] = df["Volume"].rolling(params["volume_window"]).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"].replace(0, np.nan)
    # 下落中に出来高増加 → 底値での売り圧力ピーク（反転前兆）
    df["volume_on_down_day"] = (
        (df["return_1d"] < 0) & (df["volume_ratio"] > 1.5)
    ).astype(int)
    # 上昇中に出来高増加 → 反転確認
    df["volume_on_up_day"] = (
        (df["return_1d"] > 0) & (df["volume_ratio"] > 1.3)
    ).astype(int)

    # --- ローソク足パターン（ハンマー足 = 下ヒゲ長い反転シグナル） ---
    body = abs(df["Close"] - df["Open"])
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
    total_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["lower_shadow_ratio"] = lower_shadow / total_range
    # ハンマー足: 下ヒゲが全体の60%以上、実体が小さい
    df["hammer_candle"] = (
        (df["lower_shadow_ratio"] > 0.6) & (body < total_range * 0.3)
    ).astype(int)

    # --- 複合売られすぎスコア（0-4点） ---
    df["oversold_composite"] = (
        df["rsi_oversold"] +
        df["stoch_oversold"] +
        df["williams_oversold"] +
        df["mfi_oversold"]
    )

    # --- 複合反転スコア（0-4点） ---
    df["reversal_composite"] = (
        df["stoch_cross_up"] +
        df["mfi_turning_up"] +
        df["obv_rising"] +
        df["rsi_turning"]
    )

    return df


def get_value_feature_columns() -> list[str]:
    """バリュー戦略で使用する特徴量カラムリスト。"""
    return [
        # 割安度
        "dist_from_52w_low", "dist_from_52w_high", "pct_in_52w_range",
        "bb_pct", "bb_lower_dist",
        # 売られすぎ
        "rsi", "rsi_oversold", "rsi_turning",
        "stoch_k", "stoch_d", "stoch_oversold", "stoch_cross_up",
        "williams_r", "williams_oversold",
        "mfi", "mfi_oversold", "mfi_turning_up",
        # 買い圧力
        "obv_ma_diff", "obv_rising",
        # トレンド
        "ma_short_dev", "ma_mid_dev", "in_downtrend",
        # リターン
        "return_1d", "return_3d", "return_5d", "return_1d_positive",
        # 出来高
        "volume_ratio", "volume_on_down_day", "volume_on_up_day",
        # ローソク足
        "lower_shadow_ratio", "hammer_candle",
        # 複合スコア
        "oversold_composite", "reversal_composite",
    ]


# ------------------------------------------------------------------ #
#  特徴量データセット構築
# ------------------------------------------------------------------ #

def build_value_dataset(
    price_data: dict[str, pd.DataFrame],
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """複数銘柄のデータからバリュー戦略特徴量データセットを構築する。"""
    if params is None:
        params = DEFAULT_VALUE_PARAMS

    all_dfs = []
    for ticker, df in price_data.items():
        if len(df) < params["min_rows"]:
            logger.warning(f"Skipping {ticker}: only {len(df)} rows")
            continue

        feat_df = calc_value_features(df, params)

        # 特徴量カラムのNaN行を除去（ウォームアップ期間）
        fund_cols = set()  # バリュー戦略にはファンダメンタルカラムなし
        required_cols = [
            c for c in get_value_feature_columns()
            if c in feat_df.columns and c not in fund_cols
        ]
        feat_df.dropna(subset=required_cols, inplace=True)

        if feat_df.empty:
            continue

        # Closeを残しておく（ラベリングに使用）
        feat_df["ticker"] = ticker
        all_dfs.append(feat_df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined.index.name = "date"
    combined = combined.reset_index().set_index(["ticker", "date"])
    logger.info(f"Value dataset: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


# ------------------------------------------------------------------ #
#  ラベリング・学習
# ------------------------------------------------------------------ #

def create_value_labels(dataset: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """5営業日後に10%以上上昇したかのラベルを生成する。"""
    labels = {}
    for ticker in dataset.index.get_level_values("ticker").unique():
        try:
            close = dataset.loc[ticker, "Close"].sort_index()
        except KeyError:
            continue
        future_return = close.shift(-horizon) / close - 1
        label = (future_return >= threshold).astype(float)
        label[future_return.isna()] = np.nan
        for date, val in label.items():
            labels[(ticker, date)] = val

    result = pd.Series(labels, name="label")
    result.index.names = ["ticker", "date"]
    pos_rate = result.dropna().mean()
    logger.info(
        f"Value labels: {result.notna().sum()} samples, "
        f"positive rate {pos_rate:.1%} ({horizon}d / {threshold:.0%})"
    )
    return result


def train_value_model(
    dataset: pd.DataFrame,
    params: Optional[dict] = None,
    plot: bool = True,
) -> tuple:
    """
    バリュー戦略モデルの学習・CVと最終モデル保存を行う。

    Returns: (model, scaler, cv_results)
    """
    if params is None:
        params = DEFAULT_VALUE_PARAMS

    # ラベル生成
    labels = create_value_labels(dataset, params["label_horizon"], params["label_threshold"])

    # データ準備
    feat_cols = get_value_feature_columns()
    avail = [c for c in feat_cols if c in dataset.columns]
    X = dataset[avail].copy().join(labels, how="inner").dropna(subset=["label"])
    y = X.pop("label").astype(int)
    X = X.fillna(0)

    logger.info(f"Value training: {len(X)} samples, class dist: {y.value_counts().to_dict()}")

    if len(X) < 50:
        raise ValueError(f"Training samples too few: {len(X)}")

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=params["cv_n_splits"])
    fold_results, fi_list = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params["lgb_params"])
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "fold": fold + 1,
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob) if y_val.nunique() > 1 else 0.5,
        }
        fold_results.append(metrics)
        fi_list.append(pd.Series(model.feature_importances_, index=X.columns))
        logger.info(f"Value Fold {fold+1}: Prec={metrics['precision']:.3f}, AUC={metrics['roc_auc']:.3f}")

    df_cv = pd.DataFrame(fold_results)
    print("\n" + "=" * 50)
    print("Value Strategy CV Results")
    print("=" * 50)
    for col in ["precision", "recall", "roc_auc"]:
        print(f"  {col:<12}: {df_cv[col].mean():.3f} ± {df_cv[col].std():.3f}")
    print("=" * 50)

    fi = pd.concat(fi_list, axis=1).mean(axis=1).sort_values(ascending=False)

    if plot:
        _plot_value_fi(fi)

    # 最終モデル
    Path(params["model_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    final_model = lgb.LGBMClassifier(**params["lgb_params"])
    final_model.fit(X_scaled, y)

    joblib.dump(final_model, params["model_save_path"])
    joblib.dump(scaler, params["scaler_save_path"])
    logger.info(f"Value model saved: {params['model_save_path']}")

    return final_model, scaler, {"fold_results": fold_results, "feature_importance": fi}


def _plot_value_fi(fi: pd.Series) -> None:
    """特徴量重要度プロット。"""
    top = fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top.values, y=top.index, palette="coolwarm", ax=ax)
    ax.set_title("Value Strategy — Feature Importance (Top 20)")
    plt.tight_layout()
    Path("./models").mkdir(parents=True, exist_ok=True)
    plt.savefig("./models/value_feature_importance.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------ #
#  スクリーニング
# ------------------------------------------------------------------ #

def run_value_screener(
    price_data: dict[str, pd.DataFrame],
    model_path: str = "./models/value_model.pkl",
    scaler_path: str = "./models/value_scaler.pkl",
    params: Optional[dict] = None,
    top_n: int = 20,
    output_dir: str = "./results",
) -> pd.DataFrame:
    """
    バリュー戦略: 学習済みモデルで直近データをスコアリングし結果を返す。

    2段階スクリーニング:
      Step 1: MLモデルで全銘柄をスコアリング
      Step 2: 売られすぎ複合スコアが高い銘柄を優先表示
    """
    if params is None:
        params = DEFAULT_VALUE_PARAMS

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 直近データで特徴量を計算
    dataset = build_value_dataset(price_data, params)
    if dataset.empty:
        return pd.DataFrame()

    feat_cols = get_value_feature_columns()
    avail = [c for c in feat_cols if c in dataset.columns]

    # 各銘柄の最新行を抽出
    latest_rows = []
    for ticker in dataset.index.get_level_values("ticker").unique():
        row = dataset.loc[ticker, avail].iloc[-1]
        row.name = ticker
        latest_rows.append(row)

    X_latest = pd.DataFrame(latest_rows).fillna(0)
    X_scaled = pd.DataFrame(
        scaler.transform(X_latest),
        index=X_latest.index,
        columns=X_latest.columns,
    )

    proba = model.predict_proba(X_scaled)[:, 1]
    result = pd.DataFrame({"rebound_probability": proba}, index=X_latest.index)

    # シグナル指標を付加
    sig_cols = [
        "rsi", "stoch_k", "williams_r", "mfi",
        "dist_from_52w_low", "pct_in_52w_range",
        "oversold_composite", "reversal_composite",
        "volume_ratio", "stoch_cross_up", "mfi_turning_up",
    ]
    for col in sig_cols:
        if col in X_latest.columns:
            result[col] = X_latest[col]

    # 株価情報
    for ticker, df in price_data.items():
        if ticker in result.index and not df.empty:
            result.loc[ticker, "latest_close"] = df["Close"].iloc[-1]

    result = result.sort_values("rebound_probability", ascending=False)

    # 表示・保存
    _display_value_results(result, top_n)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(f"{output_dir}/value_screening_latest.csv")

    return result


def _display_value_results(result: pd.DataFrame, top_n: int) -> None:
    from datetime import datetime
    print("\n" + "=" * 70)
    print(f"  VALUE REBOUND SCREENER — Top {top_n}")
    print(f"  Target: +10% within 5 trading days | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    cols = ["rebound_probability", "latest_close", "rsi", "stoch_k",
            "pct_in_52w_range", "oversold_composite", "reversal_composite"]
    avail = [c for c in cols if c in result.columns]
    top = result.head(top_n)[avail].copy()
    top["rebound_probability"] = top["rebound_probability"].map("{:.1%}".format)
    if "latest_close" in top.columns:
        top["latest_close"] = top["latest_close"].map("{:,.0f}".format)
    if "rsi" in top.columns:
        top["rsi"] = top["rsi"].map("{:.1f}".format)
    if "stoch_k" in top.columns:
        top["stoch_k"] = top["stoch_k"].map("{:.1f}".format)
    if "pct_in_52w_range" in top.columns:
        top["pct_in_52w_range"] = top["pct_in_52w_range"].map("{:.1%}".format)

    print(top.to_string())
    print("=" * 70)
