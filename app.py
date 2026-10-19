# -*- coding: utf-8 -*-
"""A 股双轨投机雷达 · Streamlit 入口。

本地：streamlit run app.py
手机：部署到 GitHub 后用 Streamlit Community Cloud 打开同一 URL。
"""
from __future__ import annotations

import html
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec_radar.config import (  # noqa: E402
    DEFAULT_MACRO_PATH,
    DEFAULT_MODEL_PATH,
    ScanSettings,
    load_macro_playbook,
    load_model_config,
)
from spec_radar.datasource import CN_TZ, china_now  # noqa: E402
from spec_radar.pipeline import run_scan  # noqa: E402
from spec_radar.scoring import COMPONENT_LABELS, COMPONENT_NOTES  # noqa: E402

st.set_page_config(
    page_title="A股双轨投机雷达",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_COLORS = {"potential": "#d97706", "active": "#c2410c"}
MODEL_NAMES = {"potential": "潜在投机", "active": "活跃投机"}


# --------------------------------------------------------------------------- 样式
def inject_css() -> None:
    st.markdown(
        """
<style>
  .block-container {max-width: 1240px; padding-top: 1.1rem; padding-bottom: 3rem;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  h1 {font-size: clamp(1.55rem, 4vw, 2.25rem) !important; margin-bottom: .15rem !important;
      line-height: 1.35 !important; overflow: visible !important;}
  .hero-sub {color: #5b6472; font-size: .95rem; margin: .2rem 0 .8rem 0;}
  .hero-badges {display:flex; flex-wrap:wrap; gap:.4rem; margin:.25rem 0 .8rem 0;}
  .badge {display:inline-flex; align-items:center; gap:.25rem; background:#fff7ed;
          border:1px solid #fed7aa; color:#9a3412; border-radius:999px;
          padding:.22rem .6rem; font-size:.78rem; white-space:nowrap;}
  .badge.gray {background:#f3f4f6; border-color:#e5e7eb; color:#4b5563;}
  .badge.green {background:#ecfdf5; border-color:#a7f3d0; color:#065f46;}
  .card-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));
              gap:.75rem; margin:.55rem 0 1rem 0;}
  .pick-card {background:#fff; border:1px solid #eceef2; border-radius:16px; padding:.9rem 1rem;
              box-shadow:0 2px 10px rgba(17,24,39,.045); position:relative; overflow:hidden;}
  .pick-card:before {content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
                     background:linear-gradient(#f59e0b,#c2410c);}
  .pick-card .pc-top {display:flex; justify-content:space-between; align-items:flex-start; gap:.5rem;}
  .pick-card .pc-name {font-size:1.02rem; font-weight:700; line-height:1.25;}
  .pick-card .pc-code {color:#6b7280; font-size:.76rem; font-weight:500;}
  .pick-card .pc-score {font-size:1.35rem; font-weight:800; color:#9a3412; line-height:1;}
  .pick-card .pc-rank {font-size:.7rem; color:#9ca3af; text-align:right;}
  .pick-card .pc-theme {color:#92400e; background:#fffbeb; border-radius:8px; padding:.28rem .45rem;
                        margin:.5rem 0; font-size:.76rem; line-height:1.35;}
  .pick-card .pc-metrics {display:grid; grid-template-columns:repeat(3,1fr); gap:.25rem; margin-top:.4rem;}
  .pc-metric {font-size:.72rem; color:#6b7280;}
  .pc-metric b {display:block; color:#111827; font-size:.86rem;}
  .pick-card .pc-risk {font-size:.7rem; color:#9a3412; margin-top:.5rem; line-height:1.3;
                       overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;}
  .metric-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));
                gap:.55rem; margin:.45rem 0 .9rem 0;}
  .metric-box {background:#fff; border:1px solid #eceef2; border-radius:12px; padding:.55rem .7rem;}
  .metric-box .m-label {font-size:.72rem; color:#6b7280;}
  .metric-box .m-value {font-size:1.12rem; font-weight:750; color:#111827; line-height:1.3;}
  .metric-box .m-note {font-size:.68rem; color:#9ca3af;}
  .narrative {background:#fff; border:1px solid #eceef2; border-radius:14px; padding:.8rem .9rem;
              margin-bottom:.55rem; line-height:1.65; font-size:.88rem;}
  .narrative b {color:#9a3412;}
  .detail-head {background:linear-gradient(135deg,#fff7ed,#ffffff); border:1px solid #fed7aa;
                border-radius:16px; padding:1rem 1.1rem; margin-bottom:.8rem;}
  .detail-head .dh-name {font-size:1.35rem; font-weight:800;}
  .detail-head .dh-meta {color:#6b7280; font-size:.8rem; margin-top:.15rem;}
  .detail-head .dh-theme {color:#9a3412; font-size:.8rem; margin-top:.35rem;}
  .reason {background:#fff; border-left:3px solid #f59e0b; border-radius:0 10px 10px 0;
           padding:.55rem .7rem; margin:.35rem 0; font-size:.85rem; line-height:1.55;}
  .warnbox {background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; border-radius:12px;
            padding:.6rem .8rem; font-size:.82rem; line-height:1.55;}
  .small-muted {color:#6b7280; font-size:.76rem; line-height:1.5;}
  div[data-testid="stDataFrame"] {border:1px solid #eceef2; border-radius:12px; overflow:hidden;}
  div[data-testid="stTabs"] button {font-size:.86rem; padding-left:.65rem; padding-right:.65rem;}
  div[data-testid="stMetricValue"] {font-size:1.25rem;}
  @media (max-width: 640px) {
    .block-container {padding-left:.75rem !important; padding-right:.75rem !important; padding-top:3.2rem !important;}
    section.main div[data-testid="stHorizontalBlock"], div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {
      flex-wrap: wrap !important;
    }
    section.main div[data-testid="column"], div[data-testid="stMain"] div[data-testid="column"] {
      flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important;
    }
    .card-grid {grid-template-columns:1fr;}
    .metric-grid {grid-template-columns:repeat(2,1fr);}
    .pick-card .pc-score {font-size:1.2rem;}
    h1 {font-size:1.45rem !important;}
    .hero-sub {font-size:.85rem;}
    div[data-testid="stTabs"] button {font-size:.8rem; padding-left:.45rem; padding-right:.45rem;}
    .detail-head .dh-name {font-size:1.18rem;}
  }
</style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- 缓存
@st.cache_data(ttl=1800, max_entries=4, show_spinner=False)
def cached_scan(settings_json: str, config_sig: str, session_key: str):
    # 注意：缓存函数内部不要调用任何 st.* 元素（包括进度条回调），
    # 否则第二次会话回放缓存时会触发 CacheReplayClosureError。
    raw = json.loads(settings_json)
    raw["boards"] = tuple(raw.get("boards", ("主板", "创业板", "科创板")))
    settings = ScanSettings(**raw)
    model_cfg = load_model_config()
    playbook = load_macro_playbook()
    return run_scan(settings, model_cfg=model_cfg, playbook=playbook)


def config_signature() -> str:
    parts = []
    for path in (DEFAULT_MODEL_PATH, DEFAULT_MACRO_PATH):
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


# --------------------------------------------------------------------------- 工具
def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def pct(value, digits: int = 1) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
        return f"{float(value) * 100:,.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def num(value, digits: int = 1, suffix: str = "") -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def score_color(model_key: str) -> str:
    return MODEL_COLORS.get(model_key, "#c2410c")


def cards_html(df: pd.DataFrame, model_key: str, n: int = 3) -> str:
    if df.empty:
        return "<div class='small-muted'>当前条件下没有合格标的。</div>"
    cards = []
    for _, row in df.head(n).iterrows():
        codes = f"{esc(row.get('code'))}"
        theme = esc(row.get("theme_label") or "主题映射不足")
        risk = esc(row.get("risk_flags_text") or "")
        cards.append(
            f"""
<div class="pick-card">
  <div class="pc-top">
    <div>
      <div class="pc-name">{esc(row.get('name'))}</div>
      <div class="pc-code">{codes} · {esc(row.get('industry'))}</div>
    </div>
    <div>
      <div class="pc-score">{num(row.get('score'), 1)}</div>
      <div class="pc-rank">#{int(row.get('rank'))} · 稳健度 {num(row.get('confidence'), 0)}</div>
    </div>
  </div>
  <div class="pc-theme">{theme}</div>
  <div class="pc-metrics">
    <div class="pc-metric">20日<b>{pct(row.get('ret20'))}</b></div>
    <div class="pc-metric">60日<b>{pct(row.get('ret60'))}</b></div>
    <div class="pc-metric">流通市值<b>{num(row.get('float_mcap_yi'), 0, '亿')}</b></div>
    <div class="pc-metric">20日均额<b>{num(row.get('amount20_yi'), 1, '亿')}</b></div>
    <div class="pc-metric">换手(20)<b>{num(row.get('turnover20_pct'), 1, '%')}</b></div>
    <div class="pc-metric">风险<b>{esc(row.get('risk_level'))}</b></div>
  </div>
  <div class="pc-risk">风险：{risk}</div>
</div>"""
        )
    return "<div class='card-grid'>" + "".join(cards) + "</div>"


def metric_grid(items: list[tuple[str, str, str]]) -> str:
    boxes = "".join(
        f"<div class='metric-box'><div class='m-label'>{esc(a)}</div>"
        f"<div class='m-value'>{esc(b)}</div><div class='m-note'>{esc(c)}</div></div>"
        for a, b, c in items
    )
    return f"<div class='metric-grid'>{boxes}</div>"


def table_view(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.head(int(n)).copy()
    out = pd.DataFrame({
        "#": d["rank"].astype(int),
        "代码": d["code"],
        "名称": d["name"],
        "综合分": d["score"].round(1),
        "稳健度": d["confidence"].round(0),
        "风险": d["risk_level"],
        "行业": d["industry"],
        "核心主题": d["theme_label"],
        "信号收盘": d["close_signal"].round(2),
        "20日": d["ret20"].map(lambda x: pct(x)),
        "60日": d["ret60"].map(lambda x: pct(x)),
        "流通市值(亿)": d["float_mcap_yi"].round(1),
        "20日均额(亿)": d["amount20_yi"].round(2),
        "换手20(%)": d["turnover20_pct"].round(1),
        "PE(TTM)": d["pe_ttm"].map(lambda x: num(x, 1)),
        "风险提示": d["risk_flags_text"],
    })
    return out


def export_csv(df: pd.DataFrame, model_key: str) -> bytes:
    d = df.copy()
    d["模型"] = MODEL_NAMES.get(model_key, model_key)
    rename = {
        "rank": "排名", "code": "代码", "name": "名称", "score": "综合分",
        "confidence": "稳健度", "risk_level": "风险等级", "industry": "行业",
        "theme_label": "核心主题", "close_signal": "信号日收盘", "ret20": "20日涨幅",
        "ret60": "60日涨幅", "ret120": "120日涨幅", "ret250": "250日涨幅",
        "float_mcap_yi": "流通市值(亿)", "amount20_yi": "20日均额(亿)",
        "turnover20_pct": "20日平均换手(%)", "pe_ttm": "PE(TTM)", "pb": "PB",
        "profit_yoy": "净利润同比(%)", "rev_yoy": "营收同比(%)", "roe": "ROE(%)",
        "theme_fit": "主题契合度", "concept_day": "概念当日涨幅",
        "concept_20d": "概念20日涨幅", "ind_ret20": "行业20日涨幅",
        "risk_flags_text": "风险提示", "penalty": "风险罚分", "signal_date": "信号日",
    }
    component_cols = {c: f"分项-{COMPONENT_LABELS.get(c.replace('c_', ''), c)}" for c in d.columns if c.startswith("c_")}
    rename.update(component_cols)
    for col in ("ret20", "ret60", "ret120", "ret250", "turnover20_pct", "theme_fit", "concept_day", "concept_20d", "ind_ret20"):
        if col in d:
            d[col] = (d[col] * 100).round(2)
    cols = [c for c in rename if c in d.columns] + ["模型"]
    out = d[cols].rename(columns=rename)
    return out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def plot_price(row: pd.Series, kline: pd.DataFrame) -> go.Figure:
    code = str(row["code"])
    data = kline[kline["code"].astype(str) == code].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").tail(180)
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.76, 0.24], subplot_titles=("", ""),
    )
    fig.add_trace(
        go.Candlestick(
            x=data["date"], open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            increasing_line_color="#d9480f", increasing_fillcolor="#f97316",
            decreasing_line_color="#2b8a3e", decreasing_fillcolor="#4ade80",
            name="K线",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data["date"], y=data["ma20"], name="MA20", line=dict(color="#f59e0b", width=1.4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data["date"], y=data["ma60"], name="MA60", line=dict(color="#2563eb", width=1.4)),
        row=1, col=1,
    )
    colors = np.where(data["close"] >= data["open"], "#f97316", "#4ade80")
    fig.add_trace(
        go.Bar(x=data["date"], y=data["volume"], marker_color=colors, name="成交量", opacity=0.75),
        row=2, col=1,
    )
    fig.update_layout(
        height=450, margin=dict(l=8, r=8, t=16, b=8),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False)
    return fig


def plot_radar(row: pd.Series, model_key: str) -> go.Figure:
    order = {
        "potential": ["policy", "compression", "accumulation", "base", "fundamental", "neglect", "elasticity", "regime"],
        "active": ["momentum", "theme", "volume", "breakout", "flow", "elasticity", "fundamental", "regime"],
    }[model_key]
    labels, values = [], []
    for key in order:
        col = f"c_{key}"
        if col in row.index and pd.notna(row[col]):
            labels.append(COMPONENT_LABELS.get(key, key))
            values.append(float(row[col]))
    if not labels:
        return go.Figure()
    labels.append(labels[0])
    values.append(values[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        line_color=score_color(model_key), fillcolor="rgba(249,115,22,.18)",
    ))
    fig.update_layout(
        height=360, margin=dict(l=35, r=35, t=35, b=20),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9), gridcolor="#e5e7eb"),
                   angularaxis=dict(tickfont=dict(size=10))),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font=dict(size=10),
    )
    return fig


def reason_text(row: pd.Series, model_key: str) -> list[str]:
    comps = sorted(
        [(c.replace("c_", ""), float(row[c])) for c in row.index if c.startswith("c_") and pd.notna(row[c])],
        key=lambda x: x[1], reverse=True,
    )
    top = "、".join(f"{COMPONENT_LABELS.get(k, k)} {v:.0f}分" for k, v in comps[:3]) if comps else "—"
    if model_key == "potential":
        lines = [
            f"<b>为什么是“潜在”：</b>当前不是追涨股票，而是筛选尚未充分交易、但具备产业催化与筹码沉淀的标的。主要得分来自 {esc(top)}。",
            f"<b>位置：</b>距 52 周高点 {pct(row.get('dd52'))}，距 60 日高点 {pct(row.get('dist_high60'))}；MA120 {esc('上方' if row.get('above_ma120', 0) > 0 else '下方')}，MA250 {esc('上方' if row.get('above_ma250', 0) > 0 else '下方')}。",
            f"<b>筹码：</b>20 日平均换手 {num(row.get('turnover20_pct'), 2, '%')}，20 日均额 {num(row.get('amount20_yi'), 2, '亿')}；CMF20 {num(row.get('cmf20'), 3)}，OBV 斜率（归一） {num(row.get('obv_slope20'), 3)}。",
            f"<b>压缩：</b>ATR14/价格 {num(row.get('atr14'), 3)}，60 日波幅 {num(row.get('range60'), 3)}，20日/60日量能比 {num(row.get('amount_ratio_20_60'), 2)}。",
            f"<b>主题：</b>{esc(row.get('theme_label') or '主题映射不足')}；行业 20 日 {pct(row.get('ind_ret20'))}，个股相对行业 {pct(row.get('rs20'))}。",
        ]
    else:
        lines = [
            f"<b>为什么是“活跃”：</b>股票已经被资金选择，模型寻找题材、量能、趋势与资金流的共振，同时惩罚过热。主要得分来自 {esc(top)}。",
            f"<b>动量：</b>5日 {pct(row.get('ret5'))}，20日 {pct(row.get('ret20'))}，60日 {pct(row.get('ret60'))}；MA20 斜率 {num(row.get('ma20_slope'), 3)}，RSI14 {num(row.get('rsi14'), 1)}。",
            f"<b>量能：</b>20 日均额 {num(row.get('amount20_yi'), 2, '亿')}，20 日平均换手 {num(row.get('turnover20_pct'), 2, '%')}，5日/20日量能 {num(row.get('amount_ratio_5_20'), 2)}。",
            f"<b>位置：</b>距 60 日高点 {pct(row.get('dist_high60'))}，52周位置 {num(row.get('pos52'), 2)}；突破20日 {esc('是' if row.get('breakout20', 0) > 0 else '否')}，突破60日 {esc('是' if row.get('breakout60', 0) > 0 else '否')}。",
            f"<b>主题：</b>{esc(row.get('theme_label') or '主题映射不足')}；概念 20 日 {pct(row.get('concept_20d'))}，行业相对强度 {pct(row.get('rs20'))}。",
        ]
    return lines


def render_detail(result: dict, n_potential: int, n_active: int) -> None:
    pot, act = result["potential"], result["active"]
    kline = result["kline"]
    records = {}
    for key, df, nmax in (("potential", pot, n_potential), ("active", act, n_active)):
        for _, row in df.head(nmax).iterrows():
            code = str(row["code"])
            records.setdefault(code, {"name": row["name"], "models": []})
            records[code]["models"].append((key, int(row["rank"])))
    if not records:
        st.info("没有可查看的标的。")
        return
    options = list(records)
    def label(code: str) -> str:
        info = records[code]
        tags = " / ".join(f"{MODEL_NAMES[k]}#{r}" for k, r in info["models"])
        return f"{code} {info['name']} · {tags}"
    sel = st.selectbox("选择个股", options, format_func=label, index=0)
    models = [k for k, _ in records[sel]["models"]]
    model_key = models[0]
    if len(models) > 1:
        choice = st.radio("按哪个模型查看", models, horizontal=True, format_func=lambda k: MODEL_NAMES[k])
        model_key = choice
    df = pot if model_key == "potential" else act
    hit = df[df["code"].astype(str) == sel]
    if hit.empty:
        st.warning("该标的在当前模型中不可用。")
        return
    row = hit.iloc[0]
    st.markdown(
        f"""<div class="detail-head">
        <div class="dh-name">{esc(row['name'])} <span class="pc-code">{esc(row['code'])}</span></div>
        <div class="dh-meta">{esc(row['industry'])} · {esc(row['board'])} · 信号日 {esc(row.get('signal_date'))} ·
        综合分 {num(row['score'])}（{MODEL_NAMES[model_key]} #{int(row['rank'])}）· 稳健度 {num(row['confidence'], 0)} · 风险 {esc(row['risk_level'])}</div>
        <div class="dh-theme">主题：{esc(row.get('theme_label') or '主题映射不足')}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_grid([
            ("信号收盘", num(row.get("close_signal"), 2), "前复权"),
            ("20日涨幅", pct(row.get("ret20")), "相对强度 " + pct(row.get("rs20"))),
            ("60日涨幅", pct(row.get("ret60")), "距60日高 " + pct(row.get("dist_high60"))),
            ("流通市值", num(row.get("float_mcap_yi"), 1, "亿"), "20日均额 " + num(row.get("amount20_yi"), 2, "亿")),
            ("换手(20)", num(row.get("turnover20_pct"), 2, "%"), "RSI14 " + num(row.get("rsi14"), 1)),
            ("PE(TTM)", num(row.get("pe_ttm"), 1), "PB " + num(row.get("pb"), 2)),
            ("净利同比", pct(row.get("profit_yoy")), "营收同比 " + pct(row.get("rev_yoy"))),
            ("风险罚分", num(row.get("penalty"), 1), "稳健度 " + num(row.get("confidence"), 0)),
        ]),
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1.45], gap="medium")
    with c1:
        st.markdown("#### 分项画像")
        st.plotly_chart(plot_radar(row, model_key), width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("#### 价格与均线")
        st.plotly_chart(plot_price(row, kline), width="stretch", config={"displayModeBar": False})
    st.markdown("#### 模型解释")
    for line in reason_text(row, model_key):
        st.markdown(f"<div class='reason'>{line}</div>", unsafe_allow_html=True)
    st.caption("财务字段为最新报告期。极端同比（例如 ±数百%）通常来自低基数、重组或非经常性损益，必须回看公告与财报，不能只看百分比。")
    st.markdown("#### 风险与观察位")
    risks = row.get("risk_flags") or []
    if risks:
        st.markdown(
            "<div class='warnbox'>⚠️ " + esc("；".join(risks)) + "<br/>"
            "风险标签来自量价/财务硬规则，不代表完整风险清单；公告、减持、质押、监管与流动性风险需另行核查。</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='warnbox'>模型未触发硬性风险标签，但这只说明“没有命中规则”，"
            "不代表低风险。投机仓位的止损与仓位管理仍是第一位的。</div>",
            unsafe_allow_html=True,
        )
    low60 = np.nan
    try:
        low60 = float(row["close_signal"]) / (1.0 + float(row["dist_low60"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    st.markdown(
        metric_grid([
            ("20日均线", num(row.get("ma20"), 2), "多空观察线"),
            ("60日均线", num(row.get("ma60"), 2), "趋势观察线"),
            ("60日低点", num(low60, 2), "风险参考位"),
            ("60日高点", num(float(row.get("close_signal")) / (1 + float(row.get("dist_high60"))) if pd.notna(row.get("dist_high60")) and float(row.get("dist_high60")) > -1 else np.nan, 2), "前高/套牢区"),
        ]),
        unsafe_allow_html=True,
    )
    st.caption("“观察位”只是模型计算出的价格参考，不是买卖指令。")
    with st.expander("查看分项贡献与原始特征"):
        comp_rows = []
        for key in (["policy", "compression", "accumulation", "base", "fundamental", "neglect", "elasticity", "regime"]
                    if model_key == "potential" else
                    ["momentum", "theme", "volume", "breakout", "flow", "elasticity", "fundamental", "regime"]):
            col = f"c_{key}"
            if col in row:
                comp_rows.append({
                    "分项": COMPONENT_LABELS.get(key, key),
                    "得分": round(float(row[col]), 1),
                    "说明": COMPONENT_NOTES.get(key, ""),
                })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")
        raw_cols = ["bars", "close_signal", "ret5", "ret20", "ret60", "ret120", "ret250", "atr14", "range60",
                    "vol_ratio_20_120", "cmf20", "obv_slope20", "amount_ratio_20_120", "dd52", "pos52",
                    "turnover_mean20", "amount_mean20", "beta60", "theme_fit", "ind_ret20", "rs20", "concept_day", "concept_60d"]
        raw = pd.DataFrame([{k: row.get(k) for k in raw_cols if k in row}]).T
        raw.columns = ["值"]
        st.dataframe(raw, width="stretch")


def render_macro(result: dict, playbook: dict) -> None:
    macro = result["macro"]
    st.markdown("### 市场状态与宏观情景")
    st.markdown(
        metric_grid([
            ("市场状态", str(macro.get("label")), "风险偏好 " + num(macro.get("risk_appetite"), 0)),
            ("趋势分", num(macro.get("trend_score"), 2), "指数站上均线比例"),
            ("60日宽度", pct(macro.get("breadth_60")), "当日宽度 " + pct(macro.get("breadth_day"))),
            ("波动环境", str(macro.get("vol_level")), "波动分位 " + num(macro.get("vol_percentile"), 2)),
            ("风格", str(macro.get("style")), "小盘-大盘 " + pct(macro.get("small_large"))),
            ("外部环境", str(macro.get("external")), "黄金 " + pct(macro.get("gold20")) + " / 纳指 " + pct(macro.get("nasdaq20"))),
        ]),
        unsafe_allow_html=True,
    )
    if macro.get("override_note"):
        st.caption(macro["override_note"] + "；风险偏好已按情景覆盖。")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("#### 指数趋势")
        idx = macro.get("index_table", pd.DataFrame()).copy()
        if not idx.empty:
            idx["5日"] = idx["5日"].map(pct); idx["20日"] = idx["20日"].map(pct); idx["60日"] = idx["60日"].map(pct)
            idx["20日年化波动"] = idx["20日年化波动"].map(lambda x: pct(x, 1))
            st.dataframe(idx[["名称", "收盘", "20日", "60日", "站上MA20", "站上MA60"]], hide_index=True, width="stretch")
        else:
            st.info("指数数据不可用。")
    with c2:
        st.markdown("#### 全球/风格代理")
        pxy = macro.get("proxy_table", pd.DataFrame()).copy()
        if not pxy.empty:
            pxy["20日"] = pxy["20日"].map(pct); pxy["60日"] = pxy["60日"].map(pct)
            st.dataframe(pxy[["名称", "收盘", "20日", "60日"]], hide_index=True, width="stretch")
        else:
            st.info("代理资产数据不可用。")

    st.markdown("### 政策与产业主线（人工研判底稿）")
    st.caption(f"配置基准日：{playbook.get('as_of', '—')}。这部分是研究假设，可在 config/macro_playbook.json 中修改；模型只把主题 tilt 作为可解释的暴露，而不是收益预测。")
    narrative = playbook.get("narrative", {})
    for label, text in [("世界格局", narrative.get("world")), ("国内政策", narrative.get("china_policy")),
                        ("经济与盈利", narrative.get("economy")), ("市场状态", narrative.get("market"))]:
        if text:
            st.markdown(f"<div class='narrative'><b>{esc(label)}</b><br/>{esc(text)}</div>", unsafe_allow_html=True)

    st.markdown("### 主题温度")
    theme_rows = []
    theme_returns = macro.get("theme_returns", {}) or {}
    for theme in playbook.get("themes", []):
        theme_rows.append({
            "主题": theme.get("name"),
            "政策倾向": num(theme.get("tilt"), 2),
            "代理资产20日": pct(theme_returns.get(theme.get("id"), 0.0)),
            "逻辑": theme.get("rationale", ""),
        })
    if theme_rows:
        st.dataframe(pd.DataFrame(theme_rows), hide_index=True, width="stretch")

    sources = playbook.get("sources", [])
    if sources:
        with st.expander("研判来源与数据说明"):
            for s in sources:
                st.markdown(f"- [{s.get('title','来源')}]({s.get('url','')}) · {s.get('publisher','')} · {s.get('date','')}")
            st.caption("外部链接内容仅作为宏观研判底稿，不代表程序实时抓取新闻或自动交易建议。")

    st.markdown("### 模型方法与风控")
    model_cfg = load_model_config()
    for key, label in (("potential", "潜在投机模型"), ("active", "活跃投机模型")):
        with st.expander(label + " · 权重与定义", expanded=(key == "potential")):
            m = model_cfg[key]
            st.write(m.get("definition", ""))
            rows = []
            for ck, w in m.get("weights", {}).items():
                rows.append({"分项": COMPONENT_LABELS.get(ck, ck), "权重": f"{w*100:.0f}%", "特征说明": COMPONENT_NOTES.get(ck, "")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        """
<div class="warnbox">
<b>关键约束：</b>两个模型都只使用信号日及之前的行情；潜在模型硬性排除已连续涨停、换手过热、短期涨幅过大的股票；活跃模型要求价格站上 MA20、20 日动量为正、换手不低于 1.5%，并惩罚追高。
数据来自东方财富公开接口（腾讯 K 线备用），存在延迟、缺失、复权与口径误差；财务字段是最新报告期，不是点时数据。任何排名都不构成投资建议。
</div>
        """,
        unsafe_allow_html=True,
    )
    render_validation()


def render_validation() -> None:
    st.markdown("### 价量核心子模型验证")
    path = ROOT / "data" / "validation.json"
    if not path.exists():
        st.info("尚未生成验证报告。运行 `python scripts/run_validation.py` 后会把结果写入 data/validation.json。")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"验证报告读取失败：{exc}")
        return
    st.caption(
        f"样本期 {data.get('potential',{}).get('period_start','—')} 至 {data.get('active',{}).get('period_end','—')}，"
        f"未来 {data.get('horizon_days',10)} 日收益，仅验证价量核心分项；"
        "不含主题/宏观/最新财务，股票池为当前候选集（存在幸存者偏差），不是完整模型的实盘准确率。"
    )
    for key, title, n in (("potential", "潜在投机 · 价量核心", 20), ("active", "活跃投机 · 价量核心", 10)):
        m = data.get(key, {})
        if not m.get("ok"):
            continue
        st.markdown(f"#### {title}（每个调仓日 Top {n}）")
        st.markdown(
            metric_grid([
                ("Rank IC 均值", num(m.get("rank_ic_mean"), 3), f"ICIR {num(m.get('icir'), 2)}"),
                ("IC 为正比例", pct(m.get("positive_ic_rate")), f"{m.get('rebalances')} 个调仓日"),
                ("Top 平均超额", pct(m.get("top_avg_excess")), f"Top 平均收益 {pct(m.get('top_avg_return'))}"),
                ("Top 胜率", pct(m.get("top_hit_rate")), f"跑赢样本均值 {pct(m.get('beat_universe_rate'))}"),
            ]),
            unsafe_allow_html=True,
        )
        detail = pd.DataFrame(m.get("detail", []))
        if not detail.empty:
            show = detail.copy()
            for col in ("top_return", "bottom_return", "universe_return", "top_excess", "hit_rate", "beat_rate"):
                if col in show:
                    show[col] = show[col].map(pct)
            show["rank_ic"] = show["rank_ic"].map(lambda x: num(x, 3))
            show = show.rename(columns={
                "date": "调仓日", "n": "样本数", "rank_ic": "Rank IC", "top_return": "Top收益",
                "bottom_return": "Bottom收益", "universe_return": "池均值", "top_excess": "Top超额",
                "hit_rate": "Top上涨比例", "beat_rate": "跑赢池均值比例",
            })
            with st.expander(f"{title} · 分调仓日明细"):
                st.dataframe(show, hide_index=True, width="stretch")
    st.caption("验证结论只用于判断因子方向与稳定性；样本短、风格单一时，历史表现不能外推。")


# --------------------------------------------------------------------------- 侧栏
def sidebar(settings: ScanSettings) -> tuple[ScanSettings, int, int, bool]:
    st.sidebar.markdown("## 📡 扫描控制")
    st.sidebar.caption("先选 N，再按需调整扫描深度。改变 N 不会重新抓行情。")
    n_pot = st.sidebar.number_input("潜在投机榜 N", min_value=3, max_value=50, value=settings.n_potential, step=1)
    n_act = st.sidebar.number_input("活跃投机榜 N", min_value=3, max_value=50, value=settings.n_active, step=1)
    depth = st.sidebar.selectbox(
        "扫描深度",
        options=["快速 · 200/200", "标准 · 300/300", "深度 · 500/500"],
        index=1,
        help="分别用于潜在榜与活跃榜的粗筛池；深度越大，覆盖越广、首次抓取越久。",
    )
    dep = {"快速 · 200/200": 200, "标准 · 300/300": 300, "深度 · 500/500": 500}[depth]
    risk_label = st.sidebar.selectbox(
        "宏观情景假设",
        options=["自动识别（推荐）", "偏进攻", "偏防御"],
        index={"auto": 0, "offensive": 1, "defensive": 2}.get(settings.risk_mode, 0),
        help="只调整市场环境分项与风险偏好，不改变量价硬指标。",
    )
    risk_mode = {"自动识别（推荐）": "auto", "偏进攻": "offensive", "偏防御": "defensive"}[risk_label]
    with st.sidebar.expander("高级筛选（会重新扫描）"):
        boards = st.multiselect(
            "板块范围", options=["主板", "创业板", "科创板"],
            default=list(settings.boards), help="北交所因流动性与制度差异默认排除。",
        )
        min_cap = st.number_input("最小流通市值（亿，0=模型默认）", min_value=0.0, max_value=2000.0, value=float(settings.min_float_mcap_yi), step=5.0)
        max_cap = st.number_input("最大流通市值（亿，0=模型默认）", min_value=0.0, max_value=5000.0, value=float(settings.max_float_mcap_yi), step=50.0)
        min_amt = st.number_input("最低20日均额（亿，0=模型默认）", min_value=0.0, max_value=50.0, value=float(settings.min_amount20_yi), step=0.5)
        excl_loss = st.checkbox("仅看当前盈利公司", value=settings.exclude_loss)
    if not boards:
        boards = ["主板", "创业板", "科创板"]
    new_settings = replace(
        settings, n_potential=int(n_pot), n_active=int(n_act),
        scan_potential=int(dep), scan_active=int(dep),
        boards=tuple(boards), risk_mode=risk_mode,
        min_float_mcap_yi=float(min_cap), max_float_mcap_yi=float(max_cap),
        min_amount20_yi=float(min_amt), exclude_loss=bool(excl_loss),
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 刷新行情（清空缓存）", width="stretch"):
        st.cache_data.clear()
        try:
            from spec_radar.cache import DiskCache
            from spec_radar.config import CACHE_DIR
            DiskCache(CACHE_DIR).clear()
        except Exception:
            pass
        st.session_state["flash"] = "缓存已清空，正在重新抓取行情。"
        st.rerun()
    st.sidebar.caption("首次扫描约 40–120 秒；云端免费实例可能更慢，之后会复用缓存。")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='small-muted'>⚠️ 本工具仅用于量化研究与风险识别，不构成投资建议。"
        "“投机”意味着高波动、高回撤与流动性风险，任何排名都可能失效。</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("模型 2026.10-r1 · 数据：东方财富 / 腾讯 / 新浪")
    return new_settings, int(n_pot), int(n_act), bool(excl_loss)


# --------------------------------------------------------------------------- 主逻辑
def main() -> None:
    inject_css()
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
  <div>
    <h1>📡 A股双轨投机雷达</h1>
    <div class="hero-sub">全 A 股扫描 · 潜在投机 / 活跃投机双模型 · 严格使用信号日及之前数据</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    try:
        model_cfg = load_model_config()
        playbook = load_macro_playbook()
    except Exception as exc:  # noqa: BLE001
        st.error(f"模型配置读取失败：{exc}")
        st.stop()

    try:
        base_settings = ScanSettings()
        settings, n_pot, n_act, _ = sidebar(base_settings)
    except Exception as exc:  # noqa: BLE001
        st.error(f"筛选参数不合法：{exc}")
        st.stop()

    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash)

    # 扫描缓存不包含展示用的 N，避免用户改 N 就重抓行情。
    scan_settings = replace(settings, n_potential=50, n_active=50)
    arg_json = json.dumps(scan_settings.key(), ensure_ascii=False, sort_keys=True)
    now_cn = china_now()
    session_key = now_cn.strftime("%Y%m%d") + ("_closed" if now_cn.hour * 60 + now_cn.minute >= 15 * 60 + 10 else "_preclose")
    result = None
    try:
        with st.spinner("正在抓取行情并计算双轨模型（首次约 40–120 秒）…"):
            result = cached_scan(arg_json, config_signature(), session_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"扫描失败：{type(exc).__name__}: {exc}")
        st.markdown(
            "**排查建议**：1）确认本机/云端能访问 `push2.eastmoney.com` 与 `push2his.eastmoney.com`；"
            "2）首次运行可稍后重试；3）本地可运行 `python scripts/scan.py --n 10` 查看详细日志。"
        )
        with st.expander("错误详情"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

    meta = result["meta"]
    macro = result.get("macro", {})
    pot, act = result["potential"], result["active"]
    if meta.get("warnings"):
        with st.expander(f"数据提示（{len(meta['warnings'])}）", expanded=False):
            for w in meta["warnings"]:
                st.warning(w)
    if meta.get("stale"):
        st.warning("当前结果使用了缓存数据，请核对信号日与快照时间后再使用。")

    st.markdown(
        f"""
<div class="hero-badges">
  <span class="badge">信号基准 · {esc(meta.get('signal_date'))} 收盘</span>
  <span class="badge gray">快照 · {esc(meta.get('snapshot_time'))}</span>
  <span class="badge">市场 · {esc(macro.get('label'))}（{num(macro.get('risk_appetite'),0)}）</span>
  <span class="badge green">全市场 {int(meta.get('universe_total',0)):,} 只 · 基础域 {int(meta.get('eligible_base',0)):,}</span>
  <span class="badge gray">耗时 {num(meta.get('elapsed_sec'),1)}s · 模型 {esc(meta.get('model_version'))}</span>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='warnbox'>⚠️ 这是“投机候选排序”，不是上涨预测。排名靠前只代表在模型特征上更符合投机资金的偏好；"
        "高收益叙事往往同时意味着高波动、高回撤、公告/监管/流动性风险。请独立核查并严格控制仓位。</div>",
        unsafe_allow_html=True,
    )

    if pot.empty and act.empty:
        st.error("两个模型都没有输出，请放宽高级筛选或检查数据日期。")
        st.stop()

    tab_overview, tab_pot, tab_act, tab_detail, tab_macro = st.tabs(
        ["总览", "潜在投机榜", "活跃投机榜", "个股详情", "宏观 / 方法"]
    )

    with tab_overview:
        st.markdown("### 今日双榜 Top 3")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(f"#### 🟠 潜在投机榜 · N={n_pot}")
            st.caption("尚未充分交易 · 蓄势/吸筹 · 政策与产业催化")
            st.markdown(cards_html(pot, "potential", 3), unsafe_allow_html=True)
        with c2:
            st.markdown(f"#### 🔴 活跃投机榜 · N={n_act}")
            st.caption("已经被资金选中 · 题材/量能/趋势共振 · 过热惩罚")
            st.markdown(cards_html(act, "active", 3), unsafe_allow_html=True)
        st.markdown("#### 市场状态")
        st.markdown(
            metric_grid([
                ("市场状态", str(macro.get("label")), "风险偏好 " + num(macro.get("risk_appetite"), 0) + " / 100"),
                ("60日宽度", pct(macro.get("breadth_60")), "上涨家数占比"),
                ("波动环境", str(macro.get("vol_level")), "分位 " + num(macro.get("vol_percentile"), 2)),
                ("风格", str(macro.get("style")), "外部：" + str(macro.get("external"))),
            ]),
            unsafe_allow_html=True,
        )
        st.caption(f"合格样本：潜在 {len(pot)} 只 / 活跃 {len(act)} 只 · 数据源：{meta.get('source')}")

    with tab_pot:
        st.markdown("### 潜在投机榜")
        st.caption("定义：尚未被市场充分交易、处于蓄势或低位，但具备政策/产业催化、筹码沉淀与资金潜伏特征。硬性排除短期涨幅过大、换手过热、连续涨停的股票。")
        if pot.empty:
            st.info("当前条件下没有潜在投机标的。")
        else:
            st.markdown(cards_html(pot, "potential", min(3, n_pot)), unsafe_allow_html=True)
            st.markdown("#### 完整榜单")
            st.dataframe(
                table_view(pot, n_pot), hide_index=True, width="stretch",
                height=min(620, 42 + 36 * min(n_pot, len(pot))),
                column_config={
                    "综合分": st.column_config.ProgressColumn("综合分", min_value=0, max_value=100, format="%.1f"),
                    "稳健度": st.column_config.NumberColumn("稳健度", format="%.0f"),
                },
            )
            st.download_button(
                "⬇️ 下载潜在榜 CSV", export_csv(pot.head(n_pot), "potential"),
                file_name=f"potential_rank_{meta.get('signal_date')}.csv", mime="text/csv",
                key="dl_pot",
            )
            st.caption("综合分只在同一榜单内相对比较，不同模型之间不可直接比较。")

    with tab_act:
        st.markdown("### 活跃投机榜")
        st.caption("定义：已经被资金选中、处于题材/趋势/量能共振阶段，具备短线博弈与接力价值。要求站上 MA20、20 日动量为正、换手不低于 1.5%，并惩罚追高与过热。")
        if act.empty:
            st.info("当前防御/震荡环境下没有达到活跃模型门槛的标的，这是模型的风险控制结果。")
        else:
            st.markdown(cards_html(act, "active", min(3, n_act)), unsafe_allow_html=True)
            st.markdown("#### 完整榜单")
            st.dataframe(
                table_view(act, n_act), hide_index=True, width="stretch",
                height=min(620, 42 + 36 * min(n_act, len(act))),
                column_config={
                    "综合分": st.column_config.ProgressColumn("综合分", min_value=0, max_value=100, format="%.1f"),
                    "稳健度": st.column_config.NumberColumn("稳健度", format="%.0f"),
                },
            )
            st.download_button(
                "⬇️ 下载活跃榜 CSV", export_csv(act.head(n_act), "active"),
                file_name=f"active_rank_{meta.get('signal_date')}.csv", mime="text/csv",
                key="dl_act",
            )
            st.caption("综合分只在同一榜单内相对比较，不同模型之间不可直接比较。")

    with tab_detail:
        render_detail(result, n_pot, n_act)

    with tab_macro:
        render_macro(result, playbook)
        st.markdown("---")
        st.caption(
            f"数据诊断：快照 {meta.get('snapshot_time')} · K线 {meta.get('kline_rows'):,} 行 · "
            f"缺失 {meta.get('kline_missing')} · 备用源 {meta.get('kline_fallback')} · 缓存 {meta.get('cache_dir')}"
        )


if __name__ == "__main__":
    main()
