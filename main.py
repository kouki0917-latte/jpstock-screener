"""
main.py
=======
全体パイプラインの制御と設定管理。

使い方:
  # 学習モード（データ取得 → 特徴量計算 → モデル学習）
  python main.py --mode train

  # スクリーニングモード（学習済みモデルを使って直近銘柄をスコアリング）
  python main.py --mode screen

  # 全工程実行（学習 + スクリーニング）
  python main.py --mode all

パラメータチューニングのワークフロー:
  1. CONFIGの label_horizon/label_threshold を変更して仮説を定義
  2. python main.py --mode train で学習・評価
  3. CVスコアと特徴量重要度を確認 → 仮説が有効か判断
  4. feature_params のウィンドウサイズなどを調整して再実行
"""

import argparse
import logging
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
#  ★ 中央設定パラメータ ★
#  すべての実験パラメータをここで管理する。
#  バックテスト仮説を変えたい場合はこの辞書を編集する。
# ------------------------------------------------------------------ #

CONFIG = {
    # ========== データ取得設定 ==========
    "tickers": None,           # Noneの場合はdata_fetcher.DEFAULT_TICKERSを使用
    "train_start": "2020-01-01",
    "train_end": "2024-12-31",
    "cache_dir": "./cache",
    "use_cache": True,

    # ========== 特徴量パラメータ ==========
    "feature_params": {
        "bb_window": 20,
        "bb_std": 2.0,
        "bb_squeeze_window": 10,
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_short": 5,
        "ma_mid": 25,
        "ma_long": 75,
        "volume_window": 20,
        "atr_window": 14,
        "min_rows": 100,
    },

    # ========== モデル・ラベリング設定 ==========
    # 【仮説】: 「20営業日（約1ヶ月）後に15%以上上昇」をブレイクアウトと定義
    # 仮説を変えたい場合: label_horizon/label_thresholdを編集してtrain実行
    "label_horizon": 20,
    "label_threshold": 0.15,

    # モデル選択: "lightgbm" or "random_forest"
    "model_type": "lightgbm",

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

    "rf_params": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },

    "cv_n_splits": 5,

    # ========== 保存先設定 ==========
    "model_save_path": "./models/model.pkl",
    "scaler_save_path": "./models/scaler.pkl",
    "output_dir": "./results",

    # ========== スクリーニング設定 ==========
    "screen_lookback_days": 120,  # 直近何日分のデータで特徴量を計算するか
    "screen_top_n": 20,           # 上位何銘柄を表示するか

    # ファンダメンタルデータを取得するか（時間がかかるためFalseも可）
    "fetch_fundamentals": False,

    # グラフを表示・保存するか
    "plot": True,

    # ========== バリュー戦略（格安・割安 × 1週間10%）==========
    "value_strategy": {
        "enabled": True,
        "label_horizon": 5,        # 5営業日後（約1週間）
        "label_threshold": 0.10,   # 10%以上上昇
        "model_save_path": "./models/value_model.pkl",
        "scaler_save_path": "./models/value_scaler.pkl",
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
            "scale_pos_weight": 3,
        },
        "min_rows": 130,
        "week_high_low_window": 260,
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
    },
}


# ------------------------------------------------------------------ #
#  ロギング設定
# ------------------------------------------------------------------ #

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


# ------------------------------------------------------------------ #
#  学習パイプライン
# ------------------------------------------------------------------ #

def run_train(config: dict) -> None:
    """データ取得 → 特徴量計算 → モデル学習・評価の一連パイプライン。"""
    from data_fetcher import fetch_stock_data, fetch_fundamental_data, DEFAULT_TICKERS
    from features import build_feature_dataset
    from model import run_model_pipeline

    tickers = config["tickers"] or DEFAULT_TICKERS
    logger = logging.getLogger(__name__)

    # --- Step 1: データ取得 ---
    logger.info(f"[Train] Fetching price data for {len(tickers)} tickers...")
    price_data = fetch_stock_data(
        tickers=tickers,
        start_date=config["train_start"],
        end_date=config["train_end"],
        cache_dir=config["cache_dir"],
        use_cache=config["use_cache"],
    )

    # ファンダメンタルデータ（オプション）
    fundamental_data = None
    if config["fetch_fundamentals"]:
        logger.info("[Train] Fetching fundamental data...")
        fundamental_data = fetch_fundamental_data(list(price_data.keys()))

    # --- Step 2: 特徴量計算 ---
    logger.info("[Train] Computing features...")
    dataset = build_feature_dataset(
        price_data=price_data,
        fundamental_data=fundamental_data,
        params=config["feature_params"],
    )

    if dataset.empty:
        logger.error("Feature dataset is empty. Check data fetching.")
        return

    # Closeカラムをモデルパイプラインで使えるよう確認
    if "Close" not in dataset.columns:
        import pandas as pd
        close_series = []
        for ticker, df in price_data.items():
            s = df["Close"].copy()
            s.index = pd.MultiIndex.from_tuples(
                [(ticker, d) for d in s.index], names=["ticker", "date"]
            )
            close_series.append(s)
        if close_series:
            dataset["Close"] = pd.concat(close_series)

    # --- Step 3: モデル学習・評価 ---
    logger.info("[Train] Training model...")

    model_config = {
        "label_horizon": config["label_horizon"],
        "label_threshold": config["label_threshold"],
        "model_type": config["model_type"],
        "lgb_params": config["lgb_params"],
        "rf_params": config["rf_params"],
        "cv_n_splits": config["cv_n_splits"],
        "model_save_path": config["model_save_path"],
        "scaler_save_path": config["scaler_save_path"],
    }

    model, scaler, cv_results = run_model_pipeline(
        dataset=dataset,
        params=model_config,
        plot=config["plot"],
    )

    logger.info("[Train] Breakout model complete!")
    logger.info(f"  Mean Precision: {cv_results['mean_metrics']['precision']:.3f}")
    logger.info(f"  Mean ROC-AUC:   {cv_results['mean_metrics']['roc_auc']:.3f}")

    # --- バリュー戦略モデルの学習 ---
    if config.get("value_strategy", {}).get("enabled", False):
        logger.info("[Train] Training value strategy model (5d/10%)...")
        from strategy_value import build_value_dataset, train_value_model
        value_params = config["value_strategy"]
        value_dataset = build_value_dataset(price_data, params=value_params)
        if not value_dataset.empty:
            train_value_model(value_dataset, params=value_params, plot=config["plot"])
            logger.info("[Train] Value model complete!")


# ------------------------------------------------------------------ #
#  スクリーニングパイプライン
# ------------------------------------------------------------------ #

def run_screen(config: dict) -> None:
    """学習済みモデルを使ったスクリーニング実行。"""
    from data_fetcher import DEFAULT_TICKERS
    from screener import run_screener, filter_by_signals
    import logging

    logger = logging.getLogger(__name__)

    tickers = config["tickers"] or DEFAULT_TICKERS
    logger.info(f"[Screen] Running screener for {len(tickers)} tickers...")

    result = run_screener(
        tickers=tickers,
        model_path=config["model_save_path"],
        scaler_path=config["scaler_save_path"],
        feature_params=config["feature_params"],
        lookback_days=config["screen_lookback_days"],
        top_n=config["screen_top_n"],
        output_dir=config["output_dir"],
        plot=config["plot"],
    )

    # 2段階スクリーニング: MLスコア上位 × ルールベース条件
    if not result.empty:
        filtered = filter_by_signals(
            result,
            min_probability=0.5,
            require_volume_spike=False,  # Trueにすると出来高条件必須
            require_squeeze=False,        # Trueにするとスクイーズ解放条件必須
            min_rsi=35.0,
            max_rsi=75.0,
        )

        if not filtered.empty:
            print(f"\n  2-Stage Filter Applied: {len(filtered)} tickers remain")
            print(filtered[["breakout_probability", "rsi", "volume_ratio"]].head(10).to_string())

    logger.info("[Screen] Breakout screener complete!")

    # --- バリュー戦略スクリーニング ---
    if config.get("value_strategy", {}).get("enabled", False):
        from data_fetcher import get_latest_data
        from strategy_value import run_value_screener
        value_params = config["value_strategy"]
        logger.info("[Screen] Running value screener (5d/10%)...")
        latest_data = get_latest_data(tickers, lookback_days=config["screen_lookback_days"])
        run_value_screener(
            price_data=latest_data,
            model_path=value_params["model_save_path"],
            scaler_path=value_params["scaler_save_path"],
            params=value_params,
            top_n=config["screen_top_n"],
            output_dir=config["output_dir"],
        )
        logger.info("[Screen] Value screener complete!")

    logger.info("[Screen] All done!")


# ------------------------------------------------------------------ #
#  エントリーポイント
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Japanese Stock Breakout Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train             # モデルの学習
  python main.py --mode screen            # スクリーニング実行
  python main.py --mode all               # 学習 + スクリーニング
  python main.py --mode train --no-plot   # グラフなしで学習
  python main.py --mode screen --horizon 10 --threshold 0.10  # パラメータ上書き
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "screen", "all"],
        default="all",
        help="実行モード (default: all)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="ラベリングの期間（日）。Noneの場合はCONFIGの値を使用",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="ブレイクアウト閾値（例: 0.15 = 15%）",
    )
    parser.add_argument(
        "--model",
        choices=["lightgbm", "random_forest"],
        default=None,
        help="使用するモデルタイプ",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="グラフの表示・保存をスキップ",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="ログレベル",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # コマンドライン引数でCONFIGを上書き
    config = CONFIG.copy()

    if args.horizon is not None:
        config["label_horizon"] = args.horizon
    if args.threshold is not None:
        config["label_threshold"] = args.threshold
    if args.model is not None:
        config["model_type"] = args.model
    if args.no_plot:
        config["plot"] = False

    logger.info("=" * 60)
    logger.info("  Japanese Stock Breakout Screener")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Label: {config['label_horizon']}d / {config['label_threshold']:.0%}")
    logger.info(f"  Model: {config['model_type']}")
    logger.info("=" * 60)

    # ディレクトリの事前作成
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path("./models").mkdir(parents=True, exist_ok=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "all"):
        run_train(config)

    if args.mode in ("screen", "all"):
        run_screen(config)

    logger.info("Done.")


if __name__ == "__main__":
    main()
