# -*- coding: utf-8 -*-
"""宏观/市场状态模块。

把“世界格局、政策、经济、行业、市场”拆成两类输入：
1. 可实时计算的 market regime：指数趋势、市场宽度、波动、大小盘/成长价值风格、外部资产（黄金/纳指/港股/国债）；
2. 人工研判底稿：政策与产业主线（config/macro_playbook.json，可编辑）。

模型只把可计算指标以透明规则注入排序，人工底稿用于主题映射和解释，不直接当作收益预测。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .datasource import INDEXES, PROXIES


def _tanh_norm(value: float, scale: float) -> float:
    if not np.isfinite(value) or scale <= 0:
        return 0.0
    return float(np.tanh(value / scale))


def _stats(frame: pd.DataFrame) -> Dict[str, float]:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    close = df["close"].to_numpy(dtype=float)
    if len(close) < 5:
        return {}
    rets = np.diff(close) / np.maximum(close[:-1], 1e-12)
    ma20 = float(np.mean(close[-20:])) if len(close) >= 20 else np.nan
    ma60 = float(np.mean(close[-60:])) if len(close) >= 60 else np.nan
    vol20 = float(np.std(rets[-20:], ddof=1) * np.sqrt(252)) if len(rets) >= 20 else np.nan
    return {
        "date": pd.Timestamp(df["date"].iloc[-1]),
        "close": float(close[-1]),
        "ret5": float(close[-1] / close[-6] - 1) if len(close) > 5 else np.nan,
        "ret20": float(close[-1] / close[-21] - 1) if len(close) > 20 else np.nan,
        "ret60": float(close[-1] / close[-61] - 1) if len(close) > 60 else np.nan,
        "ma20": ma20,
        "ma60": ma60,
        "above_ma20": float(close[-1] > ma20) if np.isfinite(ma20) else np.nan,
        "above_ma60": float(close[-1] > ma60) if np.isfinite(ma60) else np.nan,
        "ma20_above_ma60": float(ma20 > ma60) if np.isfinite(ma20) and np.isfinite(ma60) else np.nan,
        "vol20": vol20,
        "vol_series": pd.Series(vol20, index=[0]),
        "rets": rets,
    }


def build_macro_context(
    market_panel: pd.DataFrame,
    snapshot: pd.DataFrame,
    playbook: Dict[str, Any],
    risk_mode: str = "auto",
) -> Dict[str, Any]:
    """生成市场状态、风格、外部环境和主题温度。"""
    warnings: List[str] = []
    index_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    stats_by_secid: Dict[str, Dict[str, float]] = {}
    lookup = {secid: name for secid, name in INDEXES + PROXIES}
    has_panel = market_panel is not None and not market_panel.empty and "secid" in getattr(market_panel, "columns", [])
    if has_panel:
        panel = market_panel.copy()
        panel["secid"] = panel["secid"].astype(str)
        for secid, group in panel.groupby("secid", sort=False):
            stats = _stats(group)
            if stats:
                stats_by_secid[secid] = stats

    for secid, name in INDEXES:
        stats = stats_by_secid.get(secid)
        if not stats:
            warnings.append(f"指数缺失：{name}")
            continue
        index_rows.append({
            "代码": secid.split(".")[-1],
            "名称": name,
            "收盘": stats["close"],
            "5日": stats["ret5"],
            "20日": stats["ret20"],
            "60日": stats["ret60"],
            "站上MA20": bool(stats["above_ma20"]),
            "站上MA60": bool(stats["above_ma60"]),
            "20日年化波动": stats["vol20"],
        })
    for secid, name in PROXIES:
        stats = stats_by_secid.get(secid)
        if not stats:
            continue
        proxy_rows.append({
            "代码": secid.split(".")[-1],
            "名称": name,
            "收盘": stats["close"],
            "5日": stats["ret5"],
            "20日": stats["ret20"],
            "60日": stats["ret60"],
            "20日年化波动": stats["vol20"],
        })
    index_df = pd.DataFrame(index_rows)
    proxy_df = pd.DataFrame(proxy_rows)

    def get_ret(secid: str, horizon: int = 20) -> float:
        stats = stats_by_secid.get(secid, {})
        value = stats.get(f"ret{horizon}", np.nan)
        return float(value) if np.isfinite(value) else 0.0

    # 指数趋势分：收盘>MA20、收盘>MA60、MA20>MA60 三项平均。
    trend_parts = []
    for secid, _ in INDEXES:
        s = stats_by_secid.get(secid)
        if not s:
            continue
        vals = [s.get("above_ma20", np.nan), s.get("above_ma60", np.nan), s.get("ma20_above_ma60", np.nan)]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if vals:
            trend_parts.append(np.mean(vals))
    trend_score = float(np.mean(trend_parts)) if trend_parts else 0.5
    index_rets = [
        stats_by_secid[secid].get("ret20", np.nan)
        for secid, _ in INDEXES
        if secid in stats_by_secid
    ]
    index_ret20 = float(np.nanmedian(index_rets)) if index_rets else 0.0
    if not np.isfinite(index_ret20):
        index_ret20 = 0.0

    # 全市场宽度：剔除 ST、停牌、北交所。
    if snapshot is not None and not snapshot.empty:
        mask = (~snapshot["is_st"]) & (~snapshot["is_suspended"]) & (snapshot["board"] != "北交所")
        base = snapshot.loc[mask]
        breadth_day = float((base["day_pct_snap"] > 0).mean()) if len(base) else 0.5
        breadth_60 = float((base["ret60_snap"] > 0).mean()) if len(base) else 0.5
    else:
        breadth_day = breadth_60 = 0.5
        warnings.append("缺少全市场快照，市场宽度按中性处理")

    # 上证指数自身的波动分位（近一年）。
    vol_pct = 0.5
    probe = market_panel[market_panel["secid"] == "1.000001"] if has_panel else pd.DataFrame()
    if not probe.empty:
        p = probe.copy().sort_values("date")
        p["ret"] = p["close"].pct_change()
        vol_series = p["ret"].rolling(20).std() * np.sqrt(252)
        valid = vol_series.dropna().tail(250)
        if len(valid) >= 20:
            vol_pct = float((valid <= valid.iloc[-1]).mean())

    small_large = get_ret("1.000852", 20) - get_ret("1.000300", 20)
    growth_value = get_ret("0.399006", 20) - get_ret("1.000300", 20)
    gold20 = get_ret("1.518880", 20)
    nasdaq20 = get_ret("1.513100", 20)
    hstech20 = get_ret("1.513130", 20)
    bond20 = get_ret("1.511010", 20)

    risk = 50.0
    risk += 30.0 * (trend_score - 0.5)
    risk += 20.0 * (breadth_60 - 0.5)
    risk += 10.0 * _tanh_norm(index_ret20, 0.05)
    risk -= 12.0 * (vol_pct - 0.5)
    risk += 6.0 * _tanh_norm(small_large, 0.05)
    risk += 5.0 * _tanh_norm(nasdaq20, 0.08)
    risk += 4.0 * _tanh_norm(hstech20, 0.08)
    risk -= 5.0 * _tanh_norm(gold20, 0.08)
    risk -= 3.0 * _tanh_norm(bond20, 0.02)
    auto_risk = float(np.clip(risk, 0, 100))

    override_note = ""
    if risk_mode == "offensive":
        risk = max(auto_risk, 64.0)
        override_note = "用户手动设定偏进攻情景"
    elif risk_mode == "defensive":
        risk = min(auto_risk, 36.0)
        override_note = "用户手动设定偏防御情景"
    else:
        risk = auto_risk

    if risk >= 62:
        label = "进攻"
    elif risk >= 45:
        label = "震荡"
    else:
        label = "防御"
    if vol_pct >= 0.75:
        vol_level = "高波动"
    elif vol_pct <= 0.30:
        vol_level = "低波动"
    else:
        vol_level = "常态波动"
    if small_large > 0.02:
        style = "小盘占优"
    elif small_large < -0.02:
        style = "大盘占优"
    else:
        style = "风格均衡"
    if growth_value > 0.02:
        style += " / 成长占优"
    elif growth_value < -0.02:
        style += " / 价值占优"
    if gold20 > 0.03 and nasdaq20 < 0.0:
        external = "外部避险"
    elif nasdaq20 > 0.03 and gold20 < 0.01:
        external = "外部风险偏好回升"
    else:
        external = "外部中性"
    if bond20 > 0.01 and gold20 > 0.02:
        external += "、利多避险资产"

    theme_returns: Dict[str, float] = {}
    for theme in playbook.get("themes", []):
        proxy_returns = [get_ret(secid, 20) for secid in theme.get("proxies", []) if secid in stats_by_secid]
        theme_returns[str(theme.get("id"))] = float(np.mean(proxy_returns)) if proxy_returns else 0.0

    regime = {
        "risk_appetite": float(risk),
        "auto_risk_appetite": auto_risk,
        "label": label,
        "trend_score": float(trend_score),
        "index_ret20": float(index_ret20),
        "breadth_day": breadth_day,
        "breadth_60": breadth_60,
        "vol_percentile": float(vol_pct),
        "vol_level": vol_level,
        "style": style,
        "small_large": float(small_large),
        "growth_value": float(growth_value),
        "external": external,
        "gold20": gold20,
        "nasdaq20": nasdaq20,
        "hstech20": hstech20,
        "bond20": bond20,
        "theme_returns": theme_returns,
        "index_table": index_df,
        "proxy_table": proxy_df,
        "warnings": warnings,
        "override_note": override_note,
        "stats_by_secid": {k: {kk: vv for kk, vv in v.items() if kk not in {"rets", "vol_series"}} for k, v in stats_by_secid.items()},
    }
    return regime


def external_risk_off_score(regime: Dict[str, Any]) -> float:
    """把外部资产表现压缩到 -1..1，正数表示避险。"""
    gold = float(regime.get("gold20", 0.0) or 0.0)
    nasdaq = float(regime.get("nasdaq20", 0.0) or 0.0)
    hstech = float(regime.get("hstech20", 0.0) or 0.0)
    bond = float(regime.get("bond20", 0.0) or 0.0)
    raw = 0.5 * _tanh_norm(gold, 0.06) + 0.25 * _tanh_norm(bond, 0.02) - 0.4 * _tanh_norm(nasdaq, 0.08) - 0.25 * _tanh_norm(hstech, 0.10)
    return float(np.clip(raw, -1, 1))
