"""
features.py
===========
特徴量エンジニアリングモジュール。

「次のキオクシア」検出のための特徴量設計思想:
1. ボリンジャーバンド スクイーズ → エクスパンション検知
   - 低ボラティリティ期間（スクイーズ）の後にブレイクアウトが発生しやすい
   - バンド幅の変化率でスクイーズ状態を定量化
2. 出来高急増（Volume Spike）
   - 機関投資家の参入は出来高に先行して表れることが多い
   - 相対的な出来高増加率で異常を検知
3. モメンタム複合スコア（RSI × MACD）
   - 単一指標での誤シグナルを複合化でフィルタリング
4. 移動平均乖離率
   - 短期トレンドの強さを定量化

全パラメータは外部のCONFIG辞書から注入可能な構造にする。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  デフォルトパラメータ（main.pyのCONFIGで上書き可能）
# ------------------------------------------------------------------ #

DEFAULT_FEATURE_PARAMS = {
    # ボリンジャーバンド
    "bb_window": 20,       # 移動平均の期間
    "bb_std": 2.0,         # バンド幅の標準偏差倍率
    "bb_squeeze_window": 10,  # スクイーズ判定のルックバック期間

    # RSI
    "rsi_window": 14,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # 移動平均
    "ma_short": 5,
    "ma_mid": 25,
    "ma_long": 75,

    # 出来高
    "volume_window": 20,   # 出来高移動平均の期間

    # ATR（真の変動幅）
    "atr_window": 14,

    # 特徴量作成に必要な最低行数
    "min_rows": 100,
}


# ------------------------------------------------------------------ #
#  個別指標計算関数
# ------------------------------------------------------------------ #

def calc_bollinger_bands(df: pd.DataFrame, window: int, std: float) -> pd.DataFrame:
    """
    ボリンジャーバンドを計算し、スクイーズ関連の特徴量を追加する。

    追加カラム:
    - bb_upper, bb_lower: 上下バンド
    - bb_width: バンド幅（= (upper - lower) / middle）
    - bb_pct: 価格のバンド内位置（0=下限, 1=上限）
    - bb_squeeze_ratio: 現在のbb_width / 過去N日の平均bb_width
      → 1.0未満でスクイーズ（圧縮状態）
    """
    indicator = ta.volatility.BollingerBands(
        close=df["Close"],
        window=window,
        window_dev=std,
        fillna=False,
    )
    df["bb_upper"] = indicator.bollinger_hband()
    df["bb_middle"] = indicator.bollinger_mavg()
    df["bb_lower"] = indicator.bollinger_lband()

    # バンド幅（正規化済み）
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

    # バンド内の相対位置（%B）
    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / band_range

    return df


def calc_bb_squeeze(df: pd.DataFrame, bb_squeeze_window: int) -> pd.DataFrame:
    """
    ボリンジャーバンドのスクイーズ比率を計算する。
    squeeze_ratio < 1.0 → 通常より狭い（スクイーズ中）
    squeeze_ratio が直近で増加 → エクスパンション開始の可能性
    """
    rolling_mean_bw = df["bb_width"].rolling(window=bb_squeeze_window).mean()
    df["bb_squeeze_ratio"] = df["bb_width"] / rolling_mean_bw.replace(0, np.nan)

    # スクイーズ後のエクスパンション検知: bb_widthの変化率
    df["bb_width_chg"] = df["bb_width"].pct_change(periods=3)

    return df


def calc_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    RSIを計算する。

    追加カラム:
    - rsi: RSI値（0-100）
    - rsi_oversold: RSI < 30（売られすぎゾーン）
    - rsi_rising: RSIが直近3日で上昇中かどうか
    """
    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=window, fillna=False)
    df["rsi"] = rsi_indicator.rsi()
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    df["rsi_rising"] = (df["rsi"].diff(3) > 0).astype(int)
    return df


def calc_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    MACDを計算する。

    追加カラム:
    - macd: MACDライン
    - macd_signal: シグナルライン
    - macd_hist: ヒストグラム（macd - signal）
    - macd_cross_up: MACDがシグナルを上抜けした日（ゴールデンクロス）
    - macd_hist_rising: ヒストグラムが3日連続で増加
    """
    macd_indicator = ta.trend.MACD(
        close=df["Close"],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
        fillna=False,
    )
    df["macd"] = macd_indicator.macd()
    df["macd_signal_line"] = macd_indicator.macd_signal()
    df["macd_hist"] = macd_indicator.macd_diff()

    # ゴールデンクロス検出（前日はmacd < signal、今日はmacd > signal）
    prev_below = df["macd"].shift(1) < df["macd_signal_line"].shift(1)
    curr_above = df["macd"] > df["macd_signal_line"]
    df["macd_cross_up"] = (prev_below & curr_above).astype(int)

    # ヒストグラムが増加中（モメンタム加速）
    df["macd_hist_rising"] = (
        (df["macd_hist"] > df["macd_hist"].shift(1)) &
        (df["macd_hist"].shift(1) > df["macd_hist"].shift(2))
    ).astype(int)

    return df


def calc_moving_averages(df: pd.DataFrame, short: int, mid: int, long: int) -> pd.DataFrame:
    """
    移動平均と乖離率を計算する。

    追加カラム:
    - ma_{short/mid/long}: 各移動平均
    - ma_short_dev: 短期移動平均からの乖離率
    - ma_mid_dev: 中期移動平均からの乖離率
    - golden_cross: 短期MAが中期MAを上抜け
    - price_above_ma_long: 長期MAより株価が上
    """
    df[f"ma_{short}"] = df["Close"].rolling(window=short).mean()
    df[f"ma_{mid}"] = df["Close"].rolling(window=mid).mean()
    df[f"ma_{long}"] = df["Close"].rolling(window=long).mean()

    # 乖離率（%）
    df["ma_short_dev"] = (df["Close"] - df[f"ma_{short}"]) / df[f"ma_{short}"].replace(0, np.nan) * 100
    df["ma_mid_dev"] = (df["Close"] - df[f"ma_{mid}"]) / df[f"ma_{mid}"].replace(0, np.nan) * 100

    # ゴールデンクロス
    prev_below = df[f"ma_{short}"].shift(1) < df[f"ma_{mid}"].shift(1)
    curr_above = df[f"ma_{short}"] > df[f"ma_{mid}"]
    df["golden_cross"] = (prev_below & curr_above).astype(int)

    # 長期MAとの関係
    df["price_above_ma_long"] = (df["Close"] > df[f"ma_{long}"]).astype(int)

    return df


def calc_volume_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    出来高関連の特徴量を計算する。

    追加カラム:
    - volume_ma: 出来高の移動平均
    - volume_ratio: 当日出来高 / 移動平均出来高（Volume Spike検知）
    - volume_spike: volume_ratio > 2.0（2倍以上の出来高急増）
    - volume_trend: 出来高の5日間変化率
    - price_volume_divergence: 株価下落 & 出来高増加（需給の変化）
    """
    df["volume_ma"] = df["Volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"].replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)
    df["volume_trend"] = df["Volume"].pct_change(periods=5)

    # 価格下落中に出来高増加 → 底値圏での買い集めシグナル
    price_down = df["Close"].pct_change() < 0
    volume_up = df["volume_ratio"] > 1.5
    df["price_volume_divergence"] = (price_down & volume_up).astype(int)

    return df


def calc_atr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    ATR（Average True Range）とボラティリティ関連指標を計算する。

    追加カラム:
    - atr: ATR値
    - atr_ratio: ATR / 終値（相対的なボラティリティ）
    - daily_return: 日次リターン
    - return_5d: 5日間リターン
    - return_20d: 20日間リターン
    - high_low_ratio: 高値/安値の比率（日中変動幅）
    """
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        fillna=False,
    )
    df["atr"] = atr_indicator.average_true_range()
    df["atr_ratio"] = df["atr"] / df["Close"].replace(0, np.nan)

    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(periods=5)
    df["return_20d"] = df["Close"].pct_change(periods=20)

    df["high_low_ratio"] = df["High"] / df["Low"].replace(0, np.nan) - 1

    return df


def add_fundamental_features(df: pd.DataFrame, fundamental_row: Optional[pd.Series]) -> pd.DataFrame:
    """
    ファンダメンタル指標を時系列データに結合するプレースホルダー。

    現実装では全期間に同じ値を使用（四半期ごとの更新が理想）。
    将来的には決算日をキーにした時系列ジョインに拡張予定。

    Parameters
    ----------
    fundamental_row : pd.Series or None
        fetch_fundamental_data()の1銘柄分の行
    """
    if fundamental_row is None:
        # データがない場合はNaN列を追加（モデルがNaNを扱えるため問題なし）
        for col in ["pe_ratio", "pb_ratio", "roe", "revenue_growth", "profit_margin"]:
            df[col] = np.nan
        return df

    # ファンダメンタル値を全行に同じ値で結合
    for col in ["pe_ratio", "pb_ratio", "roe", "revenue_growth", "profit_margin"]:
        df[col] = fundamental_row.get(col, np.nan)

    return df


# ------------------------------------------------------------------ #
#  メイン特徴量計算関数
# ------------------------------------------------------------------ #

def compute_features(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    fundamental_row: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    OHLCVデータから全特徴量を計算して返す。

    Parameters
    ----------
    df : pd.DataFrame
        Open, High, Low, Close, Volumeを含む日足データ
    params : dict, optional
        特徴量計算パラメータ。Noneの場合はDEFAULT_FEATURE_PARAMSを使用
    fundamental_row : pd.Series, optional
        ファンダメンタル指標の1銘柄分データ

    Returns
    -------
    pd.DataFrame
        特徴量を追加したDataFrame（元のカラムも含む）
    """
    if params is None:
        params = DEFAULT_FEATURE_PARAMS

    df = df.copy()

    if len(df) < params["min_rows"]:
        logger.warning(f"Insufficient data: {len(df)} rows (min: {params['min_rows']})")
        return pd.DataFrame()

    # --- テクニカル指標の計算 ---
    df = calc_bollinger_bands(df, params["bb_window"], params["bb_std"])
    df = calc_bb_squeeze(df, params["bb_squeeze_window"])
    df = calc_rsi(df, params["rsi_window"])
    df = calc_macd(df, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    df = calc_moving_averages(df, params["ma_short"], params["ma_mid"], params["ma_long"])
    df = calc_volume_features(df, params["volume_window"])
    df = calc_atr(df, params["atr_window"])

    # --- ファンダメンタル指標の結合 ---
    df = add_fundamental_features(df, fundamental_row)

    # --- 複合特徴量（ブレイクアウト複合スコア）---
    # RSIが50以上 & MACDがゴールデンクロス & 出来高急増
    df["breakout_composite"] = (
        (df["rsi"] > 50).astype(int) +
        df["macd_cross_up"] +
        df["volume_spike"] +
        df["golden_cross"]
    )

    # スクイーズからのエクスパンション複合スコア
    # bb_squeeze_ratio < 0.8（強いスクイーズ）の後にbb_width_chgが正転
    df["squeeze_expansion"] = (
        (df["bb_squeeze_ratio"].shift(5) < 0.8) &
        (df["bb_width_chg"] > 0.05)
    ).astype(int)

    # NaNを含む行を削除（指標計算のウォームアップ期間）
    # ファンダメンタル列は欠損が正常なので除外して判定する
    fundamental_cols = {"pe_ratio", "pb_ratio", "roe", "revenue_growth", "profit_margin"}
    feature_cols = get_feature_columns()
    required_cols = [c for c in feature_cols if c in df.columns and c not in fundamental_cols]
    df.dropna(subset=required_cols, inplace=True)

    return df


def get_feature_columns() -> list[str]:
    """
    モデルの入力として使用する特徴量カラム名のリストを返す。
    スクリーナーとモデルで同じリストを参照するための単一定義。
    """
    return [
        # ボリンジャーバンド
        "bb_width", "bb_pct", "bb_squeeze_ratio", "bb_width_chg",
        # RSI
        "rsi", "rsi_oversold", "rsi_rising",
        # MACD
        "macd_hist", "macd_cross_up", "macd_hist_rising",
        # 移動平均
        "ma_short_dev", "ma_mid_dev", "golden_cross", "price_above_ma_long",
        # 出来高
        "volume_ratio", "volume_spike", "volume_trend", "price_volume_divergence",
        # ATR・リターン
        "atr_ratio", "daily_return", "return_5d", "return_20d", "high_low_ratio",
        # 複合特徴量
        "breakout_composite", "squeeze_expansion",
        # ファンダメンタル（欠損でも可）
        "pe_ratio", "pb_ratio", "roe", "revenue_growth", "profit_margin",
    ]


def build_feature_dataset(
    price_data: dict[str, pd.DataFrame],
    fundamental_data: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    複数銘柄のデータをまとめて特徴量変換し、1つのDataFrameにまとめる。

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        fetch_stock_data()の出力形式
    fundamental_data : pd.DataFrame, optional
        fetch_fundamental_data()の出力形式
    params : dict, optional
        特徴量計算パラメータ

    Returns
    -------
    pd.DataFrame
        インデックス=(ticker, date)のMultiIndex、カラム=特徴量
    """
    all_dfs = []

    for ticker, df in price_data.items():
        # ファンダメンタル指標の取得
        fund_row = None
        if fundamental_data is not None and ticker in fundamental_data.index:
            fund_row = fundamental_data.loc[ticker]

        feat_df = compute_features(df, params=params, fundamental_row=fund_row)

        if feat_df.empty:
            logger.warning(f"Skipping {ticker}: feature computation returned empty DataFrame")
            continue

        feat_df["ticker"] = ticker
        all_dfs.append(feat_df)

    if not all_dfs:
        logger.error("No valid feature data computed.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined.index.name = "date"
    combined = combined.reset_index().set_index(["ticker", "date"])

    logger.info(f"Feature dataset: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


if __name__ == "__main__":
    import logging
    from data_fetcher import fetch_stock_data

    logging.basicConfig(level=logging.INFO)

    # 動作確認
    price_data = fetch_stock_data(
        tickers=["7203.T", "6758.T"],
        start_date="2022-01-01",
        end_date="2024-12-31",
    )
    dataset = build_feature_dataset(price_data)
    print(dataset[get_feature_columns()].describe())
