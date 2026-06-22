"""
report_generator.py
===================
スクリーニング結果をGitHub Pages用のHTMLレポートに変換するモジュール。

GitHub Actionsから呼び出される:
  python report_generator.py --input results/screening_result_latest.csv

出力: docs/index.html
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ------------------------------------------------------------------ #
#  企業名マッピング（ティッカー → 日本語企業名）
# ------------------------------------------------------------------ #

TICKER_NAMES = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "6861.T": "キーエンス",
    "9984.T": "ソフトバンクG",
    "8306.T": "三菱UFJ FG",
    "6098.T": "リクルートHD",
    "9432.T": "NTT",
    "8035.T": "東京エレクトロン",
    "4063.T": "信越化学工業",
    "7741.T": "HOYA",
    "6367.T": "ダイキン工業",
    "7974.T": "任天堂",
    "4519.T": "中外製薬",
    "2914.T": "JT",
    "6954.T": "ファナック",
    "9433.T": "KDDI",
    "4543.T": "テルモ",
    "6702.T": "富士通",
    "8411.T": "みずほFG",
    "4502.T": "武田薬品工業",
    "6501.T": "日立製作所",
    "6503.T": "三菱電機",
    "5108.T": "ブリヂストン",
    "8058.T": "三菱商事",
    "2802.T": "味の素",
    "4523.T": "エーザイ",
    "9022.T": "JR東海",
    "8031.T": "三井物産",
    "3382.T": "セブン&アイHD",
    "4307.T": "野村総研",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "7735.T": "SCREEN HD",
    "6645.T": "オムロン",
    "6756.T": "日立国際電気",
    "6963.T": "ローム",
    "7011.T": "三菱重工業",
    "6770.T": "アルプスアルパイン",
}


# ------------------------------------------------------------------ #
#  シグナル解釈ロジック
# ------------------------------------------------------------------ #

def interpret_signals(row: pd.Series) -> list[str]:
    """各シグナルの状態をテキストで解釈する。"""
    signals = []

    if pd.notna(row.get("volume_ratio")) and row["volume_ratio"] >= 2.0:
        signals.append("🔥 出来高急増")
    if pd.notna(row.get("bb_squeeze_ratio")) and row["bb_squeeze_ratio"] < 0.8:
        signals.append("⚡ BBスクイーズ中")
    if pd.notna(row.get("squeeze_expansion")) and row["squeeze_expansion"] == 1:
        signals.append("💥 スクイーズ解放")
    if pd.notna(row.get("breakout_composite")) and row["breakout_composite"] >= 3:
        signals.append("🎯 複合シグナル強")
    if pd.notna(row.get("rsi")):
        rsi = row["rsi"]
        if rsi < 35:
            signals.append(f"📉 RSI売られすぎ({rsi:.0f})")
        elif 45 <= rsi <= 65:
            signals.append(f"✅ RSI適正({rsi:.0f})")

    return signals if signals else ["—"]


def get_alert_level(prob: float) -> tuple[str, str, str]:
    """急騰確率からアラートレベル、CSS クラス、ラベルを返す。"""
    if prob >= 0.70:
        return "high", "#ef4444", "HIGH"
    elif prob >= 0.50:
        return "mid", "#f97316", "WATCH"
    else:
        return "low", "#6b7280", "—"


# ------------------------------------------------------------------ #
#  HTML生成
# ------------------------------------------------------------------ #

def build_html(df: pd.DataFrame, generated_at: str) -> str:
    """スクリーニング結果DataFrameからHTMLを生成する。"""

    # 上位20件と全件
    top20 = df.head(20)

    # 統計サマリー
    high_count = (df["breakout_probability"] >= 0.70).sum()
    watch_count = ((df["breakout_probability"] >= 0.50) & (df["breakout_probability"] < 0.70)).sum()
    total_count = len(df)
    avg_prob = df["breakout_probability"].mean()

    # テーブル行HTML
    rows_html = ""
    for rank, (idx, row) in enumerate(top20.iterrows(), 1):
        ticker = idx
        name = TICKER_NAMES.get(ticker, ticker)
        prob = row["breakout_probability"]
        _, color, level = get_alert_level(prob)
        signals = interpret_signals(row)

        close = f"¥{row['latest_close']:,.0f}" if pd.notna(row.get("latest_close")) else "—"
        rsi = f"{row['rsi']:.1f}" if pd.notna(row.get("rsi")) else "—"
        vol_ratio = f"{row['volume_ratio']:.2f}x" if pd.notna(row.get("volume_ratio")) else "—"
        bb_sq = f"{row['bb_squeeze_ratio']:.2f}" if pd.notna(row.get("bb_squeeze_ratio")) else "—"

        signal_tags = "".join(f'<span class="signal-tag">{s}</span>' for s in signals)

        bar_width = min(prob * 100, 100)
        bar_color = color

        rows_html += f"""
        <tr>
          <td class="rank">#{rank}</td>
          <td>
            <div class="ticker-code">{ticker}</div>
            <div class="ticker-name">{name}</div>
          </td>
          <td>
            <div class="prob-bar-wrap">
              <div class="prob-bar" style="width:{bar_width:.1f}%; background:{bar_color};"></div>
            </div>
            <div class="prob-label" style="color:{bar_color}; font-weight:700;">{prob:.1%}</div>
          </td>
          <td><span class="alert-badge" style="background:{bar_color};">{level}</span></td>
          <td class="mono">{close}</td>
          <td class="mono">{rsi}</td>
          <td class="mono">{vol_ratio}</td>
          <td class="mono">{bb_sq}</td>
          <td class="signals">{signal_tags}</td>
        </tr>"""

    # 全銘柄スコア（折りたたみ可能テーブル）
    all_rows_html = ""
    for idx, row in df.iterrows():
        ticker = idx
        name = TICKER_NAMES.get(ticker, ticker)
        prob = row["breakout_probability"]
        _, color, _ = get_alert_level(prob)
        close = f"¥{row['latest_close']:,.0f}" if pd.notna(row.get("latest_close")) else "—"

        all_rows_html += f"""
        <tr>
          <td>{ticker}</td>
          <td>{name}</td>
          <td style="color:{color}; font-weight:600;">{prob:.1%}</td>
          <td class="mono">{close}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>日本株ブレイクアウトスクリーナー</title>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #1a1d27;
      --surface2: #252836;
      --border: #2d3149;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #6366f1;
      --high: #ef4444;
      --mid: #f97316;
      --low: #6b7280;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Helvetica Neue', Arial, 'Hiragino Sans', 'Yu Gothic', sans-serif;
      min-height: 100vh;
      padding: 0 0 60px;
    }}

    /* ヘッダー */
    .header {{
      background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
      border-bottom: 1px solid var(--border);
      padding: 32px 24px 28px;
    }}
    .header-inner {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .header h1 {{
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header-sub {{
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 6px;
    }}
    .header-sub strong {{ color: #a5b4fc; }}

    /* サマリーカード */
    .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin: 28px 0;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    .card-label {{
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .card-value {{
      font-size: 2rem;
      font-weight: 800;
      margin-top: 4px;
    }}
    .card-value.red {{ color: #f87171; }}
    .card-value.orange {{ color: #fb923c; }}
    .card-value.purple {{ color: #a78bfa; }}
    .card-value.blue {{ color: #60a5fa; }}

    /* メインテーブル */
    .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text);
      margin: 32px 0 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    thead tr {{
      background: var(--surface2);
    }}
    thead th {{
      padding: 12px 14px;
      text-align: left;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      font-weight: 600;
      white-space: nowrap;
    }}
    tbody tr {{
      border-top: 1px solid var(--border);
      transition: background 0.15s;
    }}
    tbody tr:hover {{ background: var(--surface2); }}
    td {{
      padding: 13px 14px;
      vertical-align: middle;
    }}
    .rank {{ color: var(--text-muted); font-size: 0.8rem; font-weight: 700; width: 40px; }}
    .ticker-code {{ font-weight: 700; font-size: 0.9rem; color: #a5b4fc; }}
    .ticker-name {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }}
    .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; }}

    /* 確率バー */
    .prob-bar-wrap {{
      background: var(--surface2);
      border-radius: 4px;
      height: 6px;
      width: 100px;
      overflow: hidden;
    }}
    .prob-bar {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s;
    }}
    .prob-label {{ font-size: 0.95rem; margin-top: 4px; }}

    /* バッジ */
    .alert-badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      color: white;
      letter-spacing: 0.5px;
    }}

    /* シグナルタグ */
    .signals {{ max-width: 240px; }}
    .signal-tag {{
      display: inline-block;
      background: rgba(99,102,241,0.15);
      border: 1px solid rgba(99,102,241,0.3);
      color: #a5b4fc;
      font-size: 0.7rem;
      padding: 2px 7px;
      border-radius: 4px;
      margin: 2px 2px 2px 0;
      white-space: nowrap;
    }}

    /* 免責 */
    .disclaimer {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 20px;
      margin-top: 32px;
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.7;
    }}
    .disclaimer strong {{ color: #f87171; }}

    /* 凡例 */
    .legend {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin: 0 0 24px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-dot {{
      width: 10px; height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    /* 折りたたみ */
    details {{ margin-top: 32px; }}
    summary {{
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text);
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
      list-style: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    summary::before {{
      content: '▶';
      font-size: 0.7rem;
      color: var(--text-muted);
      transition: transform 0.2s;
    }}
    details[open] summary::before {{ transform: rotate(90deg); }}

    /* フッター */
    .footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 0.75rem;
      margin-top: 48px;
    }}

    @media (max-width: 768px) {{
      .header h1 {{ font-size: 1.3rem; }}
      .cards {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>

<!-- ヘッダー -->
<div class="header">
  <div class="header-inner">
    <h1>🚀 日本株 ブレイクアウトスクリーナー</h1>
    <div class="header-sub">
      機械学習（LightGBM）× テクニカル分析で「次のキオクシア」候補を毎日自動抽出 |
      最終更新: <strong>{generated_at}</strong>
    </div>
  </div>
</div>

<div class="container">

  <!-- サマリーカード -->
  <div class="cards">
    <div class="card">
      <div class="card-label">スクリーニング対象</div>
      <div class="card-value blue">{total_count}<span style="font-size:1rem;color:var(--text-muted);"> 銘柄</span></div>
    </div>
    <div class="card">
      <div class="card-label">HIGH ALERT (≥70%)</div>
      <div class="card-value red">{high_count}<span style="font-size:1rem;color:var(--text-muted);"> 銘柄</span></div>
    </div>
    <div class="card">
      <div class="card-label">WATCH (50–70%)</div>
      <div class="card-value orange">{watch_count}<span style="font-size:1rem;color:var(--text-muted);"> 銘柄</span></div>
    </div>
    <div class="card">
      <div class="card-label">平均ブレイクアウト確率</div>
      <div class="card-value purple">{avg_prob:.1%}</div>
    </div>
  </div>

  <!-- 凡例 -->
  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#ef4444;"></div> HIGH（≥70%）: 要注目
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#f97316;"></div> WATCH（50–70%）: 監視継続
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#6b7280;"></div> —（<50%）: 様子見
    </div>
  </div>

  <!-- メインテーブル -->
  <div class="section-title">📊 Top 20 — ブレイクアウト候補銘柄</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>銘柄</th>
          <th>急騰確率</th>
          <th>Alert</th>
          <th>株価</th>
          <th>RSI</th>
          <th>出来高比</th>
          <th>BB幅比</th>
          <th>検出シグナル</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- 全銘柄スコア（折りたたみ） -->
  <details>
    <summary>全銘柄スコア一覧 ({total_count}銘柄)</summary>
    <div class="table-wrap" style="margin-top:16px;">
      <table>
        <thead>
          <tr>
            <th>ティッカー</th>
            <th>企業名</th>
            <th>急騰確率</th>
            <th>株価</th>
          </tr>
        </thead>
        <tbody>
          {all_rows_html}
        </tbody>
      </table>
    </div>
  </details>

  <!-- モデル仕様 -->
  <details>
    <summary>モデル仕様・特徴量について</summary>
    <div style="margin-top:20px; display:grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr)); gap:16px;">
      <div class="card">
        <div class="card-label" style="margin-bottom:10px;">予測ターゲット</div>
        <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.7;">
          「20営業日（約1ヶ月）後に株価が<strong style="color:var(--text);">15%以上上昇</strong>するか」を
          二値分類（1=上昇 / 0=その他）で予測。
        </p>
      </div>
      <div class="card">
        <div class="card-label" style="margin-bottom:10px;">主要特徴量</div>
        <ul style="font-size:0.82rem; color:var(--text-muted); line-height:2; list-style:none;">
          <li>📏 ボリンジャーバンド スクイーズ比率</li>
          <li>📈 RSI（14日）</li>
          <li>📊 MACD ゴールデンクロス</li>
          <li>🔥 出来高スパイク（20日平均比）</li>
          <li>🎯 複合ブレイクアウトスコア</li>
        </ul>
      </div>
      <div class="card">
        <div class="card-label" style="margin-bottom:10px;">モデル情報</div>
        <ul style="font-size:0.82rem; color:var(--text-muted); line-height:2; list-style:none;">
          <li>🤖 LightGBM (勾配ブースティング)</li>
          <li>📅 学習データ: 2020〜直近</li>
          <li>🔄 評価: TimeSeriesSplit (5-fold)</li>
          <li>⏰ 毎日 06:00 JST 自動更新</li>
        </ul>
      </div>
    </div>
  </details>

  <!-- 免責事項 -->
  <div class="disclaimer">
    <strong>⚠️ 免責事項:</strong>
    本ツールは機械学習を用いた統計的分析であり、投資勧誘を目的としたものではありません。
    表示されるスコアは過去データのパターンに基づく参考情報であり、将来の株価上昇を保証するものではありません。
    投資に関する判断は、ご自身の責任において行ってください。株式投資には元本割れのリスクがあります。
  </div>

  <div class="footer">
    Powered by Python × LightGBM × yfinance | Auto-updated daily via GitHub Actions
  </div>

</div>
</body>
</html>"""

    return html


# ------------------------------------------------------------------ #
#  エントリーポイント
# ------------------------------------------------------------------ #

def build_value_section(df_value: pd.DataFrame) -> str:
    """バリュー戦略のHTMLセクションを生成する。"""
    top20 = df_value.head(20)
    high_count = (df_value["rebound_probability"] >= 0.65).sum()
    watch_count = ((df_value["rebound_probability"] >= 0.45) &
                   (df_value["rebound_probability"] < 0.65)).sum()

    rows_html = ""
    for rank, (idx, row) in enumerate(top20.iterrows(), 1):
        ticker = idx
        name = TICKER_NAMES.get(ticker, ticker)
        prob = row["rebound_probability"]
        color = "#10b981" if prob >= 0.65 else "#f59e0b" if prob >= 0.45 else "#6b7280"
        level = "BUY" if prob >= 0.65 else "WATCH" if prob >= 0.45 else "—"

        close = f"¥{row['latest_close']:,.0f}" if pd.notna(row.get("latest_close")) else "—"
        rsi = f"{row['rsi']:.1f}" if pd.notna(row.get("rsi")) else "—"
        stoch = f"{row['stoch_k']:.1f}" if pd.notna(row.get("stoch_k")) else "—"
        mfi = f"{row['mfi']:.1f}" if pd.notna(row.get("mfi")) else "—"
        pct52 = f"{row['pct_in_52w_range']:.1%}" if pd.notna(row.get("pct_in_52w_range")) else "—"
        oversold = int(row["oversold_composite"]) if pd.notna(row.get("oversold_composite")) else 0
        reversal = int(row["reversal_composite"]) if pd.notna(row.get("reversal_composite")) else 0

        # 売られすぎスコアをドットで可視化
        oversold_dots = "●" * oversold + "○" * (4 - oversold)
        reversal_dots = "●" * reversal + "○" * (4 - reversal)

        bar_width = min(prob * 100, 100)
        rows_html += f"""
        <tr>
          <td class="rank">#{rank}</td>
          <td>
            <div class="ticker-code">{ticker}</div>
            <div class="ticker-name">{name}</div>
          </td>
          <td>
            <div class="prob-bar-wrap">
              <div class="prob-bar" style="width:{bar_width:.1f}%; background:{color};"></div>
            </div>
            <div class="prob-label" style="color:{color}; font-weight:700;">{prob:.1%}</div>
          </td>
          <td><span class="alert-badge" style="background:{color};">{level}</span></td>
          <td class="mono">{close}</td>
          <td class="mono" style="color:#f87171;">{rsi}</td>
          <td class="mono">{stoch}</td>
          <td class="mono">{mfi}</td>
          <td class="mono">{pct52}</td>
          <td class="mono" style="color:#f87171;">{oversold_dots}</td>
          <td class="mono" style="color:#34d399;">{reversal_dots}</td>
        </tr>"""

    return f"""
  <!-- バリュー戦略セクション -->
  <div style="margin-top:48px;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
      <h2 style="font-size:1.4rem; font-weight:800; color:#e2e8f0;">
        💰 格安・割安リバウンド戦略
      </h2>
      <span style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4);
                   color:#34d399; font-size:0.75rem; padding:3px 10px; border-radius:999px; font-weight:700;">
        5営業日 / +10% ターゲット
      </span>
    </div>
    <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:20px; line-height:1.7;">
      売られすぎゾーンに叩き込まれた割安株が底値から反転するタイミングを捉える。
      +10%を繰り返し積み上げる複利戦略。
      RSI・Stochastics・Williams%R・MFIの4指標が同時に売られすぎを示す銘柄を優先表示。
    </p>

    <!-- サマリー -->
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:24px;">
      <div class="card">
        <div class="card-label">BUY候補 (≥65%)</div>
        <div class="card-value" style="color:#34d399;">{high_count} <span style="font-size:1rem;color:#64748b;">銘柄</span></div>
      </div>
      <div class="card">
        <div class="card-label">WATCH (45–65%)</div>
        <div class="card-value" style="color:#fbbf24;">{watch_count} <span style="font-size:1rem;color:#64748b;">銘柄</span></div>
      </div>
      <div class="card">
        <div class="card-label">対象銘柄数</div>
        <div class="card-value" style="color:#60a5fa;">{len(df_value)} <span style="font-size:1rem;color:#64748b;">銘柄</span></div>
      </div>
    </div>

    <!-- 凡例 -->
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#10b981;"></div> BUY（≥65%）</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div> WATCH（45–65%）</div>
      <div class="legend-item">売られすぎスコア ●=検知済み ○=未検知</div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>銘柄</th><th>反転確率</th><th>判定</th>
            <th>株価</th><th>RSI</th><th>Stoch%K</th><th>MFI</th>
            <th>52週位置</th><th>売られすぎ(0-4)</th><th>反転シグナル(0-4)</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>

    <!-- 戦略解説 -->
    <details style="margin-top:20px;">
      <summary>戦略の詳細・エントリー条件</summary>
      <div style="margin-top:20px; display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px;">
        <div class="card">
          <div class="card-label" style="margin-bottom:8px;">エントリー条件（参考）</div>
          <ul style="font-size:0.82rem; color:#94a3b8; line-height:2; list-style:none;">
            <li>📉 RSI &lt; 35（売られすぎゾーン）</li>
            <li>📊 Stochastics &lt; 25 → 上向き転換</li>
            <li>💧 MFI &lt; 30（資金流出から流入へ）</li>
            <li>📦 52週レンジ下位30%以内</li>
            <li>🔥 ML反転確率 ≥ 65%</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-label" style="margin-bottom:8px;">リスク管理（推奨）</div>
          <ul style="font-size:0.82rem; color:#94a3b8; line-height:2; list-style:none;">
            <li>🛑 損切り: エントリー比 -5%</li>
            <li>🎯 利確: +10%到達で全売り</li>
            <li>⏱️ 保有期間: 最大5営業日</li>
            <li>💼 1銘柄集中投資は避ける</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-label" style="margin-bottom:8px;">複利シミュレーション</div>
          <ul style="font-size:0.82rem; color:#94a3b8; line-height:2; list-style:none;">
            <li>月2回 +10% → 年間 +85%</li>
            <li>月3回 +10% → 年間 +185%</li>
            <li>※ 手数料・税金を考慮すること</li>
            <li>※ 実績を保証するものではありません</li>
          </ul>
        </div>
      </div>
    </details>
  </div>"""


def generate_report(
    input_csv: str = "results/screening_result_latest.csv",
    output_html: str = "docs/index.html",
    value_csv: Optional[str] = "results/value_screening_latest.csv",
) -> None:
    """CSVからHTMLレポートを生成して保存する。"""
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, index_col=0)
    df = df.sort_values("breakout_probability", ascending=False)

    generated_at = datetime.now().strftime("%Y年%m月%d日 %H:%M JST")
    html = build_html(df, generated_at)

    # バリュー戦略セクションを挿入
    value_path = Path(value_csv) if value_csv else None
    if value_path and value_path.exists():
        df_value = pd.read_csv(value_csv, index_col=0)
        df_value = df_value.sort_values("rebound_probability", ascending=False)
        value_section = build_value_section(df_value)
        # メインテーブルの後・免責の前に挿入
        html = html.replace("  <!-- 免責事項 -->", value_section + "\n\n  <!-- 免責事項 -->")

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/screening_result_latest.csv")
    parser.add_argument("--output", default="docs/index.html")
    parser.add_argument("--value-input", default="results/value_screening_latest.csv")
    args = parser.parse_args()
    generate_report(args.input, args.output, args.value_input)
