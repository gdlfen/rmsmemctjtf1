# -*- coding: utf-8 -*-
"""启发式三浪结构识别。

这不是严格的艾略特波浪计数，而是把“一浪上涨 → 二浪回调 → 三浪启动/主升”
翻译成可回测的几何/量价条件，用于缩圈和风险提示。所有计算只使用信号日及之前的 K 线。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _pivots(df: pd.DataFrame, window: int = 4, min_move: float = 0.055) -> List[Dict[str, Any]]:
    """滚动高低点 + 交替过滤，返回从旧到新的摆动点。"""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    if n < window * 2 + 10:
        return []
    candidates: List[Dict[str, Any]] = []
    for i in range(window, n - window):
        seg_h = high[i - window:i + window + 1]
        seg_l = low[i - window:i + window + 1]
        if np.isfinite(seg_h).all() and high[i] >= np.nanmax(seg_h) - 1e-9:
            candidates.append({"pos": i, "price": float(high[i]), "kind": "high"})
        if np.isfinite(seg_l).all() and low[i] <= np.nanmin(seg_l) + 1e-9:
            candidates.append({"pos": i, "price": float(low[i]), "kind": "low"})
    candidates.sort(key=lambda x: x["pos"])
    out: List[Dict[str, Any]] = []
    for c in candidates:
        if not out:
            out.append(c)
            continue
        last = out[-1]
        if c["kind"] == last["kind"]:
            if (c["kind"] == "high" and c["price"] > last["price"]) or (
                c["kind"] == "low" and c["price"] < last["price"]
            ):
                out[-1] = c
            continue
        if abs(c["price"] / max(last["price"], 1e-9) - 1.0) >= min_move:
            out.append(c)
    return out


def _macd_flags(close: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(close, errors="coerce").dropna()
    if len(x) < 35:
        return {"gold": False, "above0": False, "gold_count20": 0, "hist": np.nan}
    dif = x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    recent = cross.tail(20)
    return {
        "gold": bool(recent.iloc[-1]) if len(recent) else False,
        "above0": bool(dif.iloc[-1] > 0 and dea.iloc[-1] > 0),
        "gold_count20": int(recent.sum()),
        "hist": float(hist.iloc[-1]),
    }


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    v = float(tr.tail(n).mean())
    return v if np.isfinite(v) and v > 0 else float(close.iloc[-1]) * 0.03


def detect_wave(df: pd.DataFrame) -> Dict[str, Any]:
    """识别最后一段“低-高-低”结构，判断三浪阶段并给出目标/买卖位。"""
    empty = {
        "wave_pattern_score": 0.0, "wave_stage": "非典型", "wave_stage_code": "none",
        "wave_position": np.nan, "wave1_gain": np.nan, "wave2_retrace": np.nan,
        "wave_launch_prob": np.nan, "wave_launch_window": "—", "wave_launch_days": np.nan,
        "wave_target1": np.nan, "wave_target2": np.nan, "wave_target3": np.nan, "wave_target4": np.nan,
        "wave_upside_base": np.nan, "wave_upside_bull": np.nan,
        "wave_buy_low": np.nan, "wave_buy_high": np.nan, "wave_stop": np.nan,
        "wave_sell_low": np.nan, "wave_sell_high": np.nan,
        "wave_risk_reward": np.nan, "wave_profit_score": 0.0, "wave_note": "K线不足或未形成典型 1-2 结构",
        "wave1_start": np.nan, "wave1_high": np.nan, "wave2_low": np.nan,
        "wave3_shape": "非典型", "wave3_shape_code": "none", "wave3_shape_score": 0.0,
        "wave3_valid": False, "wave3_confirm": False, "wave3_breakout_days": 0,
        "wave3_gap_count": 0, "wave3_macd_gold": False, "wave3_weekly_gold": False,
        "wave3_volume_ratio": np.nan, "wave3_factor_score": 0.0, "wave3_checklist": "",
    }
    if df is None or len(df) < 70:
        return empty
    d = df.tail(280).reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d:
            return empty
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if len(d) < 70:
        return empty
    close = d["close"].to_numpy(dtype=float)
    current = float(close[-1])
    piv = _pivots(d)
    if len(piv) < 3:
        return empty

    best = None
    # 从最近的“低-高-低”往前找，优先最近、结构完整的候选。
    for j in range(len(piv) - 1, 1, -1):
        p2, p1, p0 = piv[j], piv[j - 1], piv[j - 2]
        if not (p0["kind"] == "low" and p1["kind"] == "high" and p2["kind"] == "low"):
            continue
        if not (p0["pos"] < p1["pos"] < p2["pos"]):
            continue
        if len(d) - 1 - p2["pos"] > 75:
            continue
        p0v, p1v, p2v = float(p0["price"]), float(p1["price"]), float(p2["price"])
        if p0v <= 0 or p1v <= p0v or p2v <= 0:
            continue
        gain1 = p1v / p0v - 1.0
        retrace = (p1v - p2v) / max(p1v - p0v, 1e-9)
        if gain1 < 0.08 or not (0.12 <= retrace <= 0.82):
            continue
        if p2v < p0v * 0.97:
            continue
        if p1["pos"] - p0["pos"] < 3 or p2["pos"] - p1["pos"] < 2:
            continue
        # 当前必须在二浪低点附近/上方，否则结构已破位。
        if current < p2v * 0.96:
            continue
        length = p1v - p0v
        progress = (current - p1v) / max(length, 1e-9)
        if current < p1v * 1.01:
            stage = "第三浪启动前"
            code = "pre3"
        elif progress < 0.45:
            stage = "第三浪初期"
            code = "early3"
        elif progress < 1.25:
            stage = "第三浪主升"
            code = "main3"
        elif progress < 2.0:
            stage = "第三浪中后段"
            code = "late3"
        else:
            stage = "第三浪末端"
            code = "end3"
        # 二浪缩量、位置、突破强度。
        vol1 = float(pd.to_numeric(d.loc[p0["pos"]:p1["pos"], "volume"], errors="coerce").mean())
        vol2 = float(pd.to_numeric(d.loc[p1["pos"]:p2["pos"], "volume"], errors="coerce").mean())
        dryup = vol2 / max(vol1, 1e-9) if np.isfinite(vol1) and vol1 > 0 else 1.0
        ma20 = float(pd.to_numeric(d["close"], errors="coerce").rolling(20).mean().iloc[-1])
        ma60 = float(pd.to_numeric(d["close"], errors="coerce").rolling(60).mean().iloc[-1])
        volr = float(pd.to_numeric(d["volume"], errors="coerce").tail(5).mean() /
                     max(pd.to_numeric(d["volume"], errors="coerce").iloc[-25:-5].mean(), 1e-9))
        ma10 = float(pd.to_numeric(d["close"], errors="coerce").rolling(10).mean().iloc[-1])
        # 三浪确认：突破浪1高点后至少连续 2 日站稳；量能同步放大更可信。
        max_run = 0
        run = 0
        for i in range(p1["pos"] + 1, len(d)):
            if float(d["close"].iloc[i]) >= p1v * 1.005:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        breakout_days = int(max_run)
        max_high_since = float(pd.to_numeric(d.loc[p1["pos"]:, "high"], errors="coerce").max())
        failed = max_high_since >= p1v * 1.01 and (current < p1v * 0.995 or current < p2v * 0.97)
        if failed:
            stage, code = "失败三浪", "failed3"
        # 未回补跳空：只统计突破浪1高点之后形成的向上缺口。
        gap_count = 0
        for i in range(p1["pos"] + 1, len(d)):
            prev_high = float(d["high"].iloc[i - 1])
            if float(d["low"].iloc[i]) > prev_high * 1.003:
                if not any(float(d["low"].iloc[j]) <= prev_high for j in range(i + 1, len(d))):
                    gap_count += 1
        gap_count = int(min(gap_count, 5))
        daily_macd = _macd_flags(pd.Series(d["close"]))
        weekly_macd = {"gold": False, "above0": False, "gold_count20": 0}
        try:
            wd = d.copy()
            wd["date"] = pd.to_datetime(wd["date"])
            weekly = wd.set_index("date").resample("W-FRI").agg({"close": "last"}).dropna()
            weekly_macd = _macd_flags(weekly["close"])
        except Exception:
            pass
        wave3_vol = float(pd.to_numeric(d.loc[p1["pos"]:, "volume"], errors="coerce").mean() /
                          max(vol1, 1e-9))
        recent5 = float(current / close[-6] - 1.0) if len(close) >= 6 else 0.0
        recent10 = float(current / close[-11] - 1.0) if len(close) >= 11 else 0.0
        pct_days = pd.to_numeric(d["close"], errors="coerce").pct_change() * 100.0
        max_day = float(pct_days.tail(20).max() / 100.0) if len(pct_days) else 0.0
        channel_days = 0
        if len(d) >= 15 and np.isfinite(ma10):
            channel_days = int((pd.to_numeric(d["close"], errors="coerce").tail(15) >= ma10 * 0.98).sum())
        channel = channel_days >= 11 and current >= p1v * 0.98
        length3 = max(max_high_since - p2v, 0.0)
        wave3_not_shortest = (length3 >= length * 0.70) if code in ("early3", "main3", "late3", "end3") else True
        wave3_valid = bool(p2v > p0v * 1.005 and wave3_not_shortest and not failed)
        confirmed = bool(code != "pre3" and breakout_days >= 2 and (wave3_vol >= 1.05 or current >= p1v * 1.03))
        if failed:
            shape, shape_code, shape_score = "失败三浪", "failed3", 10.0
        elif code == "pre3":
            shape, shape_code, shape_score = "启动前待确认", "pending", 45.0
        elif max_high_since >= p1v * 1.005 and current >= p1v * 0.995 and breakout_days < 2:
            shape, shape_code, shape_score = "假突破待确认", "fake_break", 30.0
        elif gap_count >= 1:
            shape, shape_code, shape_score = "缺口驱动型", "gap", 88.0
        elif max_day >= 0.07 or recent5 >= 0.18:
            shape, shape_code, shape_score = "单浪劲升型", "impulse", 86.0
        elif channel:
            shape, shape_code, shape_score = "通道上行型", "channel", 78.0
        elif code in ("early3", "main3"):
            shape, shape_code, shape_score = "标准延长浪", "standard", 72.0
        elif code == "late3":
            shape, shape_code, shape_score = "三浪中后段", "late", 50.0
        elif code == "end3":
            shape, shape_code, shape_score = "三浪末端", "end", 35.0
        else:
            shape, shape_code, shape_score = "非典型", "none", 20.0
        checklist = "；".join([
            ("浪2不破浪1起点" if p2v > p0v * 1.005 else "浪2破浪1起点"),
            ("尚未突破浪1高点" if code == "pre3" else (f"突破后站稳{breakout_days}日" if breakout_days >= 2 else "突破未满2日")),
            (f"三浪量能{wave3_vol:.2f}倍" if np.isfinite(wave3_vol) else "量能缺失"),
            ("日线MACD金叉" if daily_macd["gold"] else ("日线MACD零轴上" if daily_macd["above0"] else "日线MACD未确认")),
            ("周线MACD零轴上/金叉" if (weekly_macd["above0"] or weekly_macd["gold"]) else "周线MACD未确认"),
            (f"未回补缺口{gap_count}个" if gap_count else "无未回补缺口"),
            ("三浪不最短" if wave3_not_shortest else "三浪长度不足"),
            "浪4重叠未验证",
        ])
        pattern = 45.0
        pattern += 13.0 if 0.12 <= gain1 <= 0.60 else 5.0
        pattern += 18.0 if 0.28 <= retrace <= 0.618 else 8.0
        pattern += 9.0 if dryup < 0.9 else 0.0
        pattern += 8.0 if current >= p1v * 0.98 else 0.0
        pattern += 5.0 if np.isfinite(ma60) and p2v >= ma60 * 0.96 else 0.0
        pattern += 4.0 if np.isfinite(ma20) and current >= ma20 else 0.0
        pattern += 4.0 if 8 <= p2["pos"] - p1["pos"] <= 65 else 0.0
        pattern += 5.0 if confirmed else 0.0
        pattern += 4.0 if gap_count else 0.0
        if not wave3_valid:
            pattern -= 18.0
        if failed:
            pattern = min(pattern, 25.0)
        pattern = float(np.clip(pattern, 0, 100))
        factor_score = (
            0.30 * pattern
            + 0.18 * shape_score
            + 0.14 * (100 if confirmed else (45 if code == "pre3" else 20))
            + 0.12 * (100 if wave3_valid else 10)
            + 0.10 * (100 if daily_macd["gold"] or daily_macd["above0"] else 30)
            + 0.10 * (100 if weekly_macd["gold"] or weekly_macd["above0"] else 30)
            + 0.06 * min(wave3_vol * 40, 100)
        )
        factor_score = float(np.clip(factor_score, 0, 100))
        # 启动概率：结构 + 当前位置 + 量能，仅作启发式概率，不是训练模型。
        launch_prob = 0.26 + 0.0038 * pattern + 0.16 * np.clip(current / p1v - 0.9, 0, 0.2) + 0.06 * np.clip(volr - 1, 0, 1.5)
        launch_prob = float(np.clip(launch_prob, 0.10, 0.88))
        wave2_days = max(1, p2["pos"] - p1["pos"])
        if code == "pre3":
            launch_days = float(np.clip(3 + 0.22 * wave2_days + 4 * (1 - min(dryup, 1.0)), 2, 20))
            window = "1-3" if launch_days <= 3 else ("4-10" if launch_days <= 10 else "11-20")
        else:
            launch_days = 0.0
            window = "已启动"
        t1 = p2v + length * 1.0
        t2 = p2v + length * 1.618
        t3 = p2v + length * 2.618
        t4 = p2v + length * 4.236
        # 已经越过旧目标时，展示“上方下一目标”，避免把低于现价的价位当卖点。
        if current >= t1 * 0.98:
            t1 = max(t1, current * 1.05)
        if current >= t2 * 0.98:
            t2 = max(t2, current * 1.08)
        t3 = max(t3, t2 * 1.05)
        t4 = max(t4, t3 * 1.02)
        upside_base = max(t1 / current - 1.0, 0.0)
        upside_bull = max(t2 / current - 1.0, 0.0)
        if code == "pre3":
            buy_low = max(p2v, ma20 * 0.965 if np.isfinite(ma20) else p2v)
            buy_high = min(p1v * 0.995, buy_low * 1.035)
            if buy_high <= buy_low:
                buy_high = buy_low * 1.02
            stop = min(p2v * 0.97, ma20 * 0.96 if np.isfinite(ma20) else p2v * 0.97)
            sell_low, sell_high = t1, t2
            note = "二浪回调末端；等待放量突破一浪高点或回踩二浪低点企稳，跌破二浪低点则结构失效。"
        elif code in ("early3", "main3"):
            buy_low = max(p1v * 0.96, ma20 * 0.97 if np.isfinite(ma20) else p1v * 0.96)
            buy_high = max(p1v * 1.025, buy_low * 1.02)
            stop = min(p1v * 0.94, p2v * 1.01)
            if current < p1v * 1.02:
                sell_low, sell_high = t1, t2
            else:
                sell_low, sell_high = max(t2, current * 1.06), max(t3, t2 * 1.02)
            note = "三浪已启动；优先等回踩一浪高点/MA20 不破，避免在加速段追高。"
        elif code == "late3":
            buy_low = p2v
            buy_high = p1v
            stop = max(p2v * 0.98, p1v * 0.88)
            sell_low, sell_high = max(t2, current * 1.03), t3
            note = "三浪中后段，赔率下降；只做回踩，不追加速，优先保护利润。"
        else:
            buy_low, buy_high = p2v, p1v
            stop = max(p2v * 0.98, p1v * 0.86)
            sell_low, sell_high = current * 1.02, current * 1.08
            note = "三浪可能接近末端，禁止追高；以减仓/观察为主。"
        risk = max((buy_low + buy_high) / 2 - stop, current * 0.01)
        reward = max(sell_low - (buy_low + buy_high) / 2, 0)
        rr = reward / risk if risk > 0 else np.nan
        profit_score = (
            0.42 * pattern
            + 0.25 * launch_prob * 100
            + 0.20 * min(upside_bull * 100, 45)
            + 0.13 * min(rr * 20, 20)
        )
        if code in ("pre3", "early3", "main3"):
            profit_score += 4.0
        candidate = {
            "wave_pattern_score": round(pattern, 1),
            "wave_stage": stage,
            "wave_stage_code": code,
            "wave_position": round(float(np.clip((current - p0v) / max(p1v - p0v, 1e-9), 0, 3)), 3),
            "wave1_gain": round(gain1, 4),
            "wave2_retrace": round(retrace, 4),
            "wave_launch_prob": round(launch_prob, 4),
            "wave_launch_window": window,
            "wave_launch_days": round(launch_days, 1),
            "wave_target1": round(t1, 2),
            "wave_target2": round(t2, 2),
            "wave_target3": round(t3, 2),
            "wave_target4": round(t4, 2),
            "wave_upside_base": round(upside_base, 4),
            "wave_upside_bull": round(upside_bull, 4),
            "wave_buy_low": round(buy_low, 2),
            "wave_buy_high": round(buy_high, 2),
            "wave_stop": round(stop, 2),
            "wave_sell_low": round(sell_low, 2),
            "wave_sell_high": round(sell_high, 2),
            "wave_risk_reward": round(float(rr), 2) if np.isfinite(rr) else np.nan,
            "wave_profit_score": round(float(np.clip(profit_score, 0, 100)), 1),
            "wave_note": note,
            "wave1_start": round(p0v, 2),
            "wave1_high": round(p1v, 2),
            "wave2_low": round(p2v, 2),
            "wave3_shape": shape,
            "wave3_shape_code": shape_code,
            "wave3_shape_score": round(shape_score, 1),
            "wave3_valid": bool(wave3_valid),
            "wave3_confirm": bool(confirmed),
            "wave3_breakout_days": int(breakout_days),
            "wave3_gap_count": int(gap_count),
            "wave3_macd_gold": bool(daily_macd["gold"]),
            "wave3_macd_above0": bool(daily_macd["above0"]),
            "wave3_weekly_gold": bool(weekly_macd["gold"] or weekly_macd["above0"]),
            "wave3_volume_ratio": round(wave3_vol, 3) if np.isfinite(wave3_vol) else np.nan,
            "wave3_factor_score": round(factor_score, 1),
            "wave3_checklist": checklist,
            "wave3_failure": bool(failed),
        }
        if best is None or candidate["wave_profit_score"] > best["wave_profit_score"]:
            best = candidate
    if best is None:
        return empty
    return best



def detect_first_wave(df: pd.DataFrame) -> Dict[str, Any]:
    """识别“长期横盘/底部抬高、尚未启动第一浪”的候选。

    第一浪启动前不是已经上涨的股票，而是：区间收敛、低点抬高、量能萎缩，
    当前接近平台上沿但尚未有效突破。输出条件买点/目标/止损，全部启发式。
    """
    empty = {
        "first_wave_pattern_score": 0.0, "first_wave_stage": "非典型",
        "first_wave_stage_code": "none", "first_wave_position": np.nan,
        "first_wave_launch_prob": np.nan, "first_wave_launch_window": "—",
        "first_wave_launch_days": np.nan, "first_wave_target1": np.nan,
        "first_wave_target2": np.nan, "first_wave_target3": np.nan,
        "first_wave_upside_base": np.nan, "first_wave_upside_bull": np.nan,
        "first_wave_buy_low": np.nan, "first_wave_buy_high": np.nan,
        "first_wave_stop": np.nan, "first_wave_sell_low": np.nan,
        "first_wave_sell_high": np.nan, "first_wave_risk_reward": np.nan,
        "first_wave_profit_score": 0.0, "first_wave_base_high": np.nan,
        "first_wave_base_low": np.nan, "first_wave_base_range": np.nan,
        "first_wave_base_days": np.nan,
        "first_wave_note": "K线不足或未形成第一浪平台结构",
    }
    if df is None or len(df) < 100:
        return empty
    d = df.tail(280).reset_index(drop=True).copy()
    for col in ("high", "low", "close", "volume"):
        if col not in d:
            return empty
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if len(d) < 100:
        return empty
    close = d["close"].to_numpy(dtype=float)
    current = float(close[-1])
    base = d.tail(60)
    base_high = float(base["high"].max())
    base_low = float(base["low"].min())
    if not np.isfinite(base_high) or not np.isfinite(base_low) or base_high <= base_low or base_low <= 0:
        return empty
    base_range = base_high / base_low - 1.0
    if base_range > 0.42:
        return empty
    # 已经明显突破平台/阶段涨幅过大，不属于“启动前”。
    if current > base_high * 1.035 or current < base_high * 0.88:
        return empty
    ret60 = current / float(close[-61]) - 1.0 if len(close) > 61 else 0.0
    if ret60 > 0.38 or ret60 < -0.25:
        return empty
    ma20 = float(pd.to_numeric(d["close"], errors="coerce").rolling(20).mean().iloc[-1])
    ma60 = float(pd.to_numeric(d["close"], errors="coerce").rolling(60).mean().iloc[-1])
    low20 = float(d["low"].tail(20).min())
    prior_low = float(d["low"].tail(60).head(40).min())
    higher_low = low20 >= prior_low * 1.01
    vol20 = float(pd.to_numeric(d["volume"], errors="coerce").tail(20).mean())
    vol60 = float(pd.to_numeric(d["volume"], errors="coerce").tail(60).mean())
    dryup = vol20 / max(vol60, 1e-9) if np.isfinite(vol20) and np.isfinite(vol60) else 1.0
    atr_ratio = _atr(d) / max(current, 1e-9)
    prox = float(np.clip(current / base_high, 0.85, 1.035))
    pattern = 44.0
    pattern += 15.0 if 0.10 <= base_range <= 0.30 else (8.0 if base_range <= 0.38 else 0.0)
    pattern += 12.0 if current >= base_high * 0.95 else (6.0 if current >= base_high * 0.90 else 0.0)
    pattern += 8.0 if higher_low else 0.0
    pattern += 8.0 if dryup < 0.9 else (4.0 if dryup < 1.05 else 0.0)
    pattern += 6.0 if atr_ratio < 0.055 else 0.0
    pattern += 5.0 if np.isfinite(ma20) and current >= ma20 else 0.0
    pattern += 4.0 if np.isfinite(ma20) and np.isfinite(ma60) and ma20 >= ma60 * 0.98 else 0.0
    pattern += 4.0 if ret60 < 0.22 else 0.0
    pattern = float(np.clip(pattern, 0, 100))
    launch_prob = 0.22 + 0.0042 * pattern + 0.14 * np.clip(prox - 0.9, 0, 0.15) + 0.07 * np.clip(1 - dryup, 0, 1)
    launch_prob = float(np.clip(launch_prob, 0.10, 0.86))
    base_days = 60
    launch_days = float(np.clip(2 + 0.07 * base_days + 5 * max(0.0, 1 - dryup) + 4 * max(0.0, 1 - prox), 2, 20))
    window = "1-3" if launch_days <= 3 else ("4-10" if launch_days <= 10 else "11-20")
    height = base_high - base_low
    t1 = max(current * 1.06, base_high + height * 0.80)
    t2 = max(t1 * 1.06, base_high + height * 1.50)
    t3 = max(t2 * 1.08, base_high + height * 2.20)
    upside_base = max(t1 / current - 1.0, 0.0)
    upside_bull = max(t2 / current - 1.0, 0.0)
    # 平台突破买点：在平台上沿附近，不在远离平台的上方追。
    buy_low = min(current * 0.985, base_high * 1.005)
    buy_high = min(current * 1.005, base_high * 1.015)
    if buy_high <= buy_low:
        buy_high = buy_low * 1.01
    stop = min(base_low * 0.97, ma20 * 0.96 if np.isfinite(ma20) else base_low * 0.97, buy_low * 0.95)
    sell_low, sell_high = t1, t2
    risk = max((buy_low + buy_high) / 2 - stop, current * 0.01)
    reward = max(sell_low - (buy_low + buy_high) / 2, 0)
    rr = reward / risk if risk > 0 else np.nan
    profit_score = (
        0.45 * pattern
        + 0.25 * launch_prob * 100
        + 0.18 * min(upside_bull * 100, 45)
        + 0.12 * min(rr * 20, 20)
    )
    stage = "第一浪临界" if current >= base_high * 0.995 else "第一浪启动前"
    code = "pre1_break" if current >= base_high * 0.995 else "pre1"
    return {
        "first_wave_pattern_score": round(pattern, 1),
        "first_wave_stage": stage,
        "first_wave_stage_code": code,
        "first_wave_position": round(float(np.clip((current - base_low) / max(base_high - base_low, 1e-9), 0, 1.2)), 3),
        "first_wave_launch_prob": round(launch_prob, 4),
        "first_wave_launch_window": window,
        "first_wave_launch_days": round(launch_days, 1),
        "first_wave_target1": round(t1, 2),
        "first_wave_target2": round(t2, 2),
        "first_wave_target3": round(t3, 2),
        "first_wave_upside_base": round(upside_base, 4),
        "first_wave_upside_bull": round(upside_bull, 4),
        "first_wave_buy_low": round(buy_low, 2),
        "first_wave_buy_high": round(buy_high, 2),
        "first_wave_stop": round(stop, 2),
        "first_wave_sell_low": round(sell_low, 2),
        "first_wave_sell_high": round(sell_high, 2),
        "first_wave_risk_reward": round(float(rr), 2) if np.isfinite(rr) else np.nan,
        "first_wave_profit_score": round(float(np.clip(profit_score, 0, 100)), 1),
        "first_wave_base_high": round(base_high, 2),
        "first_wave_base_low": round(base_low, 2),
        "first_wave_base_range": round(base_range, 4),
        "first_wave_base_days": base_days,
        "first_wave_note": "平台收敛、低点抬高/量能萎缩；放量站上平台上沿才确认第一浪启动，跌破平台下沿则结构失效。",
    }


def adjust_wave_targets_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """现价已越过信号日目标时，把展示目标上移到现价上方的下一目标。"""
    if df is None or df.empty or "wave_target1" not in df.columns:
        return df
    out = df.copy()
    cur = pd.to_numeric(out.get("current_price", out.get("close_signal")), errors="coerce")
    t1 = pd.to_numeric(out["wave_target1"], errors="coerce")
    t2 = pd.to_numeric(out.get("wave_target2"), errors="coerce")
    out["wave_target1"] = np.where(cur.notna() & t1.notna() & (cur > t1), cur * 1.03, t1)
    if "wave_target2" in out:
        out["wave_target2"] = np.maximum(
            np.where(cur.notna() & t2.notna() & (cur > t2), cur * 1.06, t2),
            pd.to_numeric(out["wave_target1"], errors="coerce") * 1.03,
        )
    if "wave_sell_low" in out:
        out["wave_sell_low"] = np.maximum(
            pd.to_numeric(out["wave_sell_low"], errors="coerce"),
            pd.to_numeric(out["wave_target1"], errors="coerce"),
        )
    if "wave_sell_high" in out and "wave_target2" in out:
        out["wave_sell_high"] = np.maximum(
            pd.to_numeric(out["wave_sell_high"], errors="coerce"),
            pd.to_numeric(out["wave_target2"], errors="coerce"),
        )
    if "wave_upside_base" in out:
        out["wave_upside_base"] = (
            pd.to_numeric(out["wave_target1"], errors="coerce") / cur - 1.0
        ).clip(lower=0)
    if "wave_upside_bull" in out and "wave_target2" in out:
        out["wave_upside_bull"] = (
            pd.to_numeric(out["wave_target2"], errors="coerce") / cur - 1.0
        ).clip(lower=0)
    return out


def build_wave_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """对 panel 中每只股票计算三浪特征，返回以 code 为索引的宽表。"""
    if panel is None or panel.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for code, g in panel.groupby("code", sort=False):
        g = g.sort_values("date")
        if len(g) < 70:
            continue
        res = detect_wave(g)
        res.update(detect_first_wave(g))
        res["code"] = str(code).zfill(6)
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("code")
