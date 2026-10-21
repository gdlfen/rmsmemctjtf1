# -*- coding: utf-8 -*-
"""把用户的“题材—盘子—位置—量价—资金—故事/安全边际”框架落成可核对打分。

这些分数独立于综合分展示，并作为标签/筛子使用：
- 潜在六维：0–2 分/维度，总分 12；≥9 进入观察池，≥11 等买点；
- 活跃六格：6 个可勾选条件；≥5 项才标注“通过检查”。

它们是“研究框架”，不是交易指令；缺失数据不会自动算作通过。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _s(row: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _clip2(value: float) -> int:
    return int(np.clip(round(value), 0, 2))


def _bucket_total(value: float) -> tuple[int, str]:
    if value >= 11:
        return int(value), "等买点"
    if value >= 9:
        return int(value), "观察池"
    return int(value), "不达标"


def add_potential_framework(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    dims: List[Dict[str, Any]] = []
    grades, intervals = [], []
    for _, row in out.iterrows():
        theme_fit = _s(row, "theme_fit", 0)
        theme_tilt = _s(row, "theme_tilt", 0)
        concept_heat = max(_s(row, "concept_day", 0), _s(row, "concept_20d", 0), _s(row, "concept_60d", 0))
        ann_red = str(row.get("ann_red_text") or "")
        if ann_red:
            theme = 0
        elif theme_fit >= 0.55 and theme_tilt >= 0.55 and (concept_heat > 0 or _s(row, "ind_ret20", 0) > 0):
            theme = 2
        elif theme_fit >= 0.30:
            theme = 1
        else:
            theme = 0

        cap = _s(row, "float_mcap_yi", 0)
        if 50 <= cap <= 120:
            chips = 2
        elif 20 <= cap <= 200:
            chips = 1
        else:
            chips = 0
        dd = _s(row, "dd52", -1)
        ret20 = _s(row, "ret20", 0)
        if -0.42 <= dd <= -0.15 and ret20 < 0.30:
            position = 2
        elif -0.55 <= dd <= -0.05 and ret20 < 0.50:
            position = 1
        else:
            position = 0

        compression = _s(row, "c_compression", 50)
        accumulation = _s(row, "c_accumulation", 50)
        amount_trend = _s(row, "amount_ratio_20_120", 1)
        if compression >= 60 and accumulation >= 60 and amount_trend <= 1.5:
            volume = 2
        elif compression >= 45 or accumulation >= 45:
            volume = 1
        else:
            volume = 0
        if amount_trend > 2.0:
            volume = max(0, volume - 1)

        cmf = _s(row, "cmf20", 0)
        obv = _s(row, "obv_slope20", 0)
        updown = _s(row, "up_down_vol_ratio", 0)
        fund_hits = int(cmf > 0.02) + int(obv > 0) + int(updown > 1.0)
        funds = 2 if fund_hits >= 3 else (1 if fund_hits >= 2 else 0)

        penalty = _s(row, "penalty", 0)
        if ann_red:
            safety = 0
        elif penalty <= 4:
            safety = 2
        elif penalty <= 12:
            safety = 1
        else:
            safety = 0
        total = theme + chips + position + volume + funds + safety
        total, grade = _bucket_total(total)
        dims.append({"题材": theme, "盘子": chips, "位置": position, "量价": volume, "资金": funds, "安全": safety})
        grades.append(grade)
        intervals.append({
            "潜在观察区": f"{_num(row.get('low250'))} – {_num(row.get('high250'))}",
            "止损参考": _num(_s(row, "ma60", np.nan)),
            "框架": "六维 0–2 分；≥9 观察，≥11 等买点",
        })
    dim_df = pd.DataFrame(dims, index=out.index)
    out["setup_score"] = dim_df.sum(axis=1).astype(int)
    out["setup_grade"] = grades
    out["setup_dims"] = dim_df.apply(lambda r: " / ".join(f"{k}{int(v)}" for k, v in r.items()), axis=1)
    out["framework_note"] = intervals
    return out


def add_active_framework(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    checks = []
    counts = []
    for _, row in out.iterrows():
        theme_fit = _s(row, "theme_fit", 0)
        sector_n = _s(row, "sector_limitup_count", 0)
        theme = bool(theme_fit >= 0.40 and (_s(row, "concept_day", 0) > 0 or _s(row, "ind_ret20", 0) > 0 or sector_n >= 2))
        cap = 30 <= _s(row, "float_mcap_yi", 0) <= 200
        board = _s(row, "consecutive_limit_up", 0)
        ret20 = _s(row, "ret20", 0)
        position = bool((1 <= board <= 5) or (ret20 < 0.30 and _s(row, "close_signal", 0) >= _s(row, "ma20", np.inf)))
        turnover = _s(row, "turnover_mean20", 0)
        amplitude = _s(row, "amplitude_today", 0)
        open_count = _s(row, "open_count", 0)
        volume = bool(turnover >= 0.10 and (amplitude >= 0.07 or board >= 1) and open_count <= 2)
        funds = bool(_s(row, "cmf20", 0) > 0.0 or _s(row, "lhb_net_yi", 0) > 0.0)
        story = bool(theme_fit >= 0.30 or str(row.get("ann_good_text") or ""))
        ann_red = str(row.get("ann_red_text") or "")
        safety = bool(not ann_red and _s(row, "penalty", 0) < 15)
        vals = [theme, cap, position, volume, funds, story]  # 六格；安全是红线闸门
        n = int(sum(vals))
        checks.append({
            "题材": theme, "盘子": cap, "位置": position, "量价": volume,
            "资金": funds, "故事": story, "安全红线": safety,
        })
        counts.append(n)
    cdf = pd.DataFrame(checks, index=out.index)
    out["check_count"] = counts
    out["check_total"] = 6
    out["check_safety"] = cdf["安全红线"].astype(bool)
    out["check_text"] = cdf.apply(lambda r: " / ".join(f"{k}{'✓' if v else '×'}" for k, v in r.items()), axis=1)
    out["check_grade"] = [
        ("通过" if n >= 5 else ("观察" if n >= 4 else "不达标")) if safe else "红线剔除"
        for n, safe in zip(counts, out["check_safety"])
    ]
    return out


def add_trade_levels(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """给出条件式计划买点/卖点，不是“必然成交价”。

    - 潜在：等待回踩支撑或放量突破，买点通常在当前价下方；
    - 活跃：等待回踩 MA10 或前高确认，避免在加速段追高；
    - 卖点：第一目标为预测目标高价下沿/前高，第二目标为上沿；
    - 止损：MA20/MA60/近期低点与固定百分比的较保守者。
    """
    if df.empty:
        return df
    out = df.copy()
    rows = []
    for _, row in out.iterrows():
        close = _s(row, "close_signal", np.nan)
        if not np.isfinite(close) or close <= 0:
            rows.append({})
            continue
        now = _s(row, "current_price", close)
        if not np.isfinite(now) or now <= 0:
            now = close
        atr = _s(row, "atr14", 0.03) * close
        ma5 = _s(row, "ma5", close)
        ma10 = _s(row, "ma10", close)
        ma20 = _s(row, "ma20", close)
        ma60 = _s(row, "ma60", close)
        dist_low60 = _s(row, "dist_low60", np.nan)
        low60 = close / (1 + dist_low60) if np.isfinite(dist_low60) and dist_low60 > -1 else close * 0.9
        dist_high60 = _s(row, "dist_high60", np.nan)
        high60 = close / (1 + dist_high60) if np.isfinite(dist_high60) and dist_high60 > -1 else close * 1.1
        t_low = _s(row, "target_high_low", close * 1.05)
        t_mid = _s(row, "target_high", close * 1.08)
        t_bull = _s(row, "target_high_bull", close * 1.15)
        if model_key == "potential":
            buy_low = max(low60, ma20 * 0.965, ma60 * 0.96)
            buy_high = min(close * 1.015, ma20 * 1.035)
            if buy_high <= buy_low:
                buy_high = buy_low * 1.02
            # 支撑位不应跑到现价远上方；否则“买点”变成追高。
            buy_high = min(buy_high, now * 1.01)
            buy_low = min(buy_low, buy_high * 0.99)
            best_buy = (buy_low + buy_high) / 2
            stop = min(buy_low * 0.97, ma60 * 0.985) if np.isfinite(ma60) and ma60 < buy_low else buy_low * 0.97
            trigger = "回踩 MA20/60 或 60 日低点缩量企稳；或放量突破 20 日高点后回踩确认"
            sell_low = max(t_low, high60 * 0.98, best_buy * 1.06, now * 1.03)
            sell_high = max(t_bull, t_mid, sell_low * 1.02)
            note = "潜在票先等回踩/突破确认，不在放量长上影当天追；破位不补仓。"
        else:
            buy_low = min(ma10, close * 0.985) * 0.995 if np.isfinite(ma10) else close * 0.96
            buy_high = max(ma10 * 1.01, buy_low * 1.015) if np.isfinite(ma10) else close * 0.99
            if buy_high >= now:
                buy_high = now * 0.995
            if buy_high <= buy_low:
                buy_high = buy_low * 1.01
            buy_high = min(buy_high, now * 1.005)
            buy_low = min(buy_low, buy_high * 0.99)
            best_buy = (buy_low + buy_high) / 2
            stop = min(ma20 * 0.97, buy_low * 0.94, close * 0.92) if np.isfinite(ma20) else buy_low * 0.94
            trigger = "回踩 MA10 不破且缩量，或突破前高后 1–2 日不破；不追 5 日加速末端"
            sell_low = max(t_low, high60 * 0.99, best_buy * 1.06, now * 1.03)
            sell_high = max(t_bull, t_mid, sell_low * 1.02)
            note = "活跃票按接力/波段处理，跌破 MA20 或放量长阴先离场。"
        stop = min(stop, best_buy * 0.99) if stop < best_buy else best_buy * 0.92
        risk = max(best_buy - stop, close * 0.01)
        reward = max(sell_low - best_buy, 0)
        rr = reward / risk if risk > 0 else np.nan
        rows.append({
            "buy_zone_low": round(float(buy_low), 2),
            "buy_zone_high": round(float(buy_high), 2),
            "best_buy_price": round(float(best_buy), 2),
            "buy_trigger": trigger,
            "stop_loss": round(float(stop), 2),
            "sell_zone_low": round(float(sell_low), 2),
            "sell_zone_high": round(float(sell_high), 2),
            "best_sell_price": round(float(sell_low), 2),
            "risk_reward": round(float(rr), 2) if np.isfinite(rr) else np.nan,
            "trade_note": note,
        })
    add = pd.DataFrame(rows, index=out.index)
    for col in add.columns:
        out[col] = add[col]
    return out


def _num(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
        return f"{v:.{digits}f}" if np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def build_daily_screens(features: pd.DataFrame, snapshot: pd.DataFrame, scored: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """生成首板池、连板池、高弹性分歧池（基于信号日特征，非实时盘中）。"""
    if features.empty:
        return {}
    df = features.copy()
    snap = snapshot.set_index("code")
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    else:
        df["code"] = df.index.astype(str).str.zfill(6)
    for col in ("name", "industry", "float_mcap_yi", "net_profit"):
        if col not in df:
            df[col] = snap[col].reindex(df["code"]).values
    common = (~snap["is_st"].reindex(df["code"]).fillna(False).values) & (df["avg_amount20_yi"].fillna(0) >= 0.45)
    first_flag = df.get("first_board", pd.Series(0.0, index=df.index)).fillna(0)
    consecutive_flag = df.get("consecutive_limit_up", pd.Series(0.0, index=df.index)).fillna(0)
    amplitude = df.get("amplitude_today", pd.Series(0.0, index=df.index)).fillna(0)
    turnover_today = df.get("turnover_today", df["turnover_mean20"]).fillna(0)
    volume_ratio = df.get("volume_ratio_5", pd.Series(0.0, index=df.index)).fillna(0)
    first = df[common & (first_flag == 1)].copy()
    consecutive = df[common & (consecutive_flag.between(2, 5))].copy()
    divergence = df[
        common
        & (df["float_mcap_yi"].fillna(0).between(30, 200))
        & (turnover_today >= 0.12)
        & (amplitude >= 0.07)
        & (df["ret20"].fillna(0) < 0.30)
        & (volume_ratio >= 1.5)
    ].copy()
    score_map = {}
    for key, frame in scored.items():
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                score_map[str(row["code"])] = (float(row["score"]), key, int(row["rank"]))
    for name, frame in (("首板启动池", first), ("连板确认池", consecutive), ("高弹性分歧池", divergence)):
        if frame.empty:
            continue
        frame["模型分"] = frame["code"].map(lambda c: score_map.get(c, (np.nan, "", np.nan))[0])
        frame["模型"] = frame["code"].map(lambda c: score_map.get(c, (np.nan, "", np.nan))[1])
        frame["模型排名"] = frame["code"].map(lambda c: score_map.get(c, (np.nan, "", np.nan))[2])
        frame.sort_values(["模型分", "consecutive_limit_up"], ascending=False, inplace=True)
        frame.reset_index(drop=True, inplace=True)
        scored[name] = frame
    return {"首板启动池": first, "连板确认池": consecutive, "高弹性分歧池": divergence}
