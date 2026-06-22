"""
data_fetcher.py
================
株価データの取得モジュール。
yfinanceを使って日本株の日足データを取得し、キャッシュに保存する。

設計方針:
- 銘柄リストは外部から注入可能（テスト・本番で差し替えやすい）
- キャッシュ機能でAPI呼び出し回数を削減
- 取得失敗した銘柄はスキップし、成功分だけ返す
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  定数
# ------------------------------------------------------------------ #

# yfinanceでの日本株ティッカーは "{証券コード}.T" の形式
NIKKEI225_TICKERS_SAMPLE = [
    "7203.T",  # トヨタ自動車
    "6758.T",  # ソニーグループ
    "6861.T",  # キーエンス
    "9984.T",  # ソフトバンクグループ
    "8306.T",  # 三菱UFJフィナンシャル
    "6098.T",  # リクルートホールディングス
    "9432.T",  # NTT
    "8035.T",  # 東京エレクトロン
    "4063.T",  # 信越化学工業
    "7741.T",  # HOYA
    "6367.T",  # ダイキン工業
    "7974.T",  # 任天堂
    "4519.T",  # 中外製薬
    "2914.T",  # JT
    "6954.T",  # ファナック
    "9433.T",  # KDDI
    "4543.T",  # テルモ
    "6702.T",  # 富士通
    "8411.T",  # みずほフィナンシャルグループ
    "4502.T",  # 武田薬品工業
    "6501.T",  # 日立製作所
    "6503.T",  # 三菱電機
    "5108.T",  # ブリヂストン
    "8058.T",  # 三菱商事
    "2802.T",  # 味の素
    "4523.T",  # エーザイ
    "9022.T",  # 東海旅客鉄道（JR東海）
    "8031.T",  # 三井物産
    "3382.T",  # セブン&アイ・ホールディングス
    "4307.T",  # 野村総合研究所
]

# キオクシア（2024年上場）のような半導体・テック銘柄を追加
SEMICONDUCTOR_TICKERS = [
    "6920.T",  # レーザーテック
    "6146.T",  # ディスコ
    "7735.T",  # SCREENホールディングス
    "6645.T",  # オムロン
    "4523.T",  # エーザイ
    "6756.T",  # 日立国際電気
    "6588.T",  # 東芝テック
    "6963.T",  # ローム
    "7011.T",  # 三菱重工業
    "6770.T",  # アルプスアルパイン
]

DEFAULT_TICKERS = list(set(NIKKEI225_TICKERS_SAMPLE + SEMICONDUCTOR_TICKERS))


# ------------------------------------------------------------------ #
#  キャッシュユーティリティ
# ------------------------------------------------------------------ #

def _cache_path(ticker: str, cache_dir: str) -> Path:
    """ティッカーに対応するキャッシュファイルパスを返す。"""
    return Path(cache_dir) / f"{ticker.replace('.', '_')}.parquet"


def _load_from_cache(ticker: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """キャッシュからデータを読み込む。存在しない場合はNoneを返す。"""
    path = _cache_path(ticker, cache_dir)
    if path.exists():
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit: {ticker}")
        return df
    return None


def _save_to_cache(df: pd.DataFrame, ticker: str, cache_dir: str) -> None:
    """データをparquet形式でキャッシュに保存する。"""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)
    df.to_parquet(path)
    logger.debug(f"Cached: {ticker}")


# ------------------------------------------------------------------ #
#  メイン取得関数
# ------------------------------------------------------------------ #

def fetch_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "./cache",
    use_cache: bool = True,
    sleep_sec: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    指定した銘柄リストの日足OHLCVデータを取得する。

    Parameters
    ----------
    tickers : list[str]
        ティッカーシンボルのリスト（例: ["7203.T", "6758.T"]）
    start_date : str
        取得開始日（例: "2020-01-01"）
    end_date : str
        取得終了日（例: "2024-12-31"）
    cache_dir : str
        キャッシュ保存先ディレクトリ
    use_cache : bool
        Trueの場合、キャッシュが存在すればAPIを叩かない
    sleep_sec : float
        API呼び出し間のスリープ時間（レートリミット対策）

    Returns
    -------
    dict[str, pd.DataFrame]
        {ティッカー: DataFrame} の辞書
        DataFrameのカラム: Open, High, Low, Close, Volume
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tqdm(tickers, desc="Fetching stock data"):
        # キャッシュチェック
        if use_cache:
            cached = _load_from_cache(ticker, cache_dir)
            if cached is not None:
                result[ticker] = cached
                continue

        # yfinanceでデータ取得
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # 株式分割・配当調整済みの価格を使用
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                failed.append(ticker)
                continue

            # カラム名をフラット化（MultiIndexになる場合があるため）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 必要なカラムのみ抽出
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.dropna(how="all", inplace=True)

            result[ticker] = df
            if use_cache:
                _save_to_cache(df, ticker, cache_dir)

            time.sleep(sleep_sec)

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"Fetched {len(result)} tickers. Failed: {failed}")
    return result


def fetch_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "./cache",
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """単一銘柄のデータを取得する便利関数。"""
    result = fetch_stock_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    return result.get(ticker)


def get_latest_data(
    tickers: list[str],
    lookback_days: int = 60,
    cache_dir: str = "./cache",
) -> dict[str, pd.DataFrame]:
    """
    スクリーニング用に直近N日のデータを取得する。
    キャッシュは使わず常に最新データを取得する。
    """
    from datetime import datetime, timedelta
    end_date = datetime.today().strftime("%Y-%m-%d")
    # バッファとして取得期間を長めにとる（祝日・土日除外のため）
    start_date = (datetime.today() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

    return fetch_stock_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        use_cache=False,  # 最新データは必ずAPIから取得
        sleep_sec=0.3,
    )


# ------------------------------------------------------------------ #
#  ファンダメンタル指標（プレースホルダー）
# ------------------------------------------------------------------ #

def fetch_fundamental_data(tickers: list[str]) -> pd.DataFrame:
    """
    ファンダメンタル指標を取得するプレースホルダー。

    現時点ではyfinanceのinfo辞書から取得可能な項目を返す。
    本番環境では以下のような代替ソースへの差し替えを想定:
    - 決算短信API（EDINET、TDnet）
    - Bloomberg / Refinitiv のAPIクライアント
    - kabutan.jp のスクレイピング

    Returns
    -------
    pd.DataFrame
        インデックス=ティッカー, カラム=各種ファンダメンタル指標
    """
    records = []

    for ticker in tqdm(tickers, desc="Fetching fundamentals"):
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                # PER（株価収益率）
                "pe_ratio": info.get("trailingPE", None),
                # PBR（株価純資産倍率）
                "pb_ratio": info.get("priceToBook", None),
                # 時価総額（円）
                "market_cap": info.get("marketCap", None),
                # ROE
                "roe": info.get("returnOnEquity", None),
                # 売上高成長率（直近）
                "revenue_growth": info.get("revenueGrowth", None),
                # 利益率
                "profit_margin": info.get("profitMargins", None),
                # 配当利回り
                "dividend_yield": info.get("dividendYield", None),
                # 浮動株比率
                "float_shares_ratio": (
                    info.get("floatShares", 0) / info.get("sharesOutstanding", 1)
                    if info.get("sharesOutstanding") else None
                ),
            })
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 動作確認: サンプル銘柄でデータ取得テスト
    test_tickers = ["7203.T", "6758.T", "8035.T"]
    data = fetch_stock_data(
        tickers=test_tickers,
        start_date="2023-01-01",
        end_date="2024-12-31",
        cache_dir="./cache",
    )
    for ticker, df in data.items():
        print(f"\n{ticker}: {len(df)} rows")
        print(df.tail(3))
