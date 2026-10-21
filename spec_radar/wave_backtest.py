# -*- coding: utf-8 -*-
"""三浪 / 第一浪策略的历史事件回测（离线灾备数据）。

口径：
- 信号日只用当日及之前 K 线；
- 次日开盘入场；
- 10 个交易日内先触止损视为止损，触目标视为目标，否则第 10 日收盘退出；
- 成本按单边 0.1% 估算；
- 当前快照仅用于市值/ST 静态过滤，历史公告/减持/立案无法完整点时重建，回测中不包含，
  实盘页面会对第一浪候选单独做公告红线核查。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .bootstrap import load_market_panel, load_stock_panel
from .indicators import technical_features
from .wave import detect_first_wave, detect_wave


def _six_proxy(feat: pd.Series, cap: float, is_st: bool) -> Dict[str, Any]:
    """六维/六格的价量代理；主题/故事用中性分，财务/公告不进入历史回测。"""
    ret20 = float(feat.get("ret20", np.nan) or 0)
    dd52 = float(feat.get("dd52", np.nan) or -0.3)
    atr = float(feat.get("atr14", np.nan) or 0.05)
    amount_ratio = float(feat.get("amount_ratio_20_120", np.nan) or 1.0)
    vcp = float(feat.get("vcp_ratio", np.nan) or 1.0)
    cmf = float(feat.get("cmf20", np.nan) or 0)
    obv = float(feat.get("obv_slope20", np.nan) or 0)
    ma20 = float(feat.get("ma20", np.nan) or np.nan)
    close = float(feat.get("close_signal", np.nan) or np.nan)
    turnover = float(feat.get("turnover_mean20", np.nan) or 0)
    amount20 = float(feat.get("avg_amount20_yi", np.nan) or 0)
    theme = 1
    size = 2 if 50 <= cap <= 120 else (1 if 20 <= cap <= 200 else 0)
    position = 2 if (-0.45 <= dd52 <= -0.12 and ret20 < 0.30) else (1 if dd52 < -0.05 else 0)
    volume = 2 if (atr < 0.06 and amount_ratio < 1.05 and vcp < 0.75) else (1 if amount_ratio < 1.3 else 0)
    funds = 2 if (cmf > 0 and obv > 0) else (1 if (cmf > 0 or obv > 0) else 0)
    safety = 2 if not is_st else 0
    setup = theme + size + position + volume + funds + safety

    active_theme = 1
    active_size = 1 if 30 <= cap <= 200 else 0
    active_pos = 1 if (ret20 >= 0 and np.isfinite(ma20) and close >= ma20) else 0
    active_volume = 1 if (amount20 >= 1.0 and turnover >= 0.015) else 0
    active_funds = 1 if (cmf > 0 or obv > 0) else 0
    active_story = 1
    active_safety = 1 if not is_st else 0
    checks = active_theme + active_size + active_pos + active_volume + active_funds + active_story + active_safety
    return {
        "setup_proxy": int(setup),
        "check_proxy": int(min(checks, 6)),
        "six_pass": bool(setup >= 9 or checks >= 5),
        "six_proxy_text": f"六维{setup}/六格{min(checks,6)}",
    }


def _universe_return(date: pd.Timestamp, groups: Dict[str, pd.DataFrame], horizon: int) -> float:
    vals = []
    for g in groups.values():
        pos = int(g["date"].searchsorted(date, side="left"))
        if pos < 0 or pos >= len(g) or g["date"].iloc[pos] != date:
            continue
        if pos + horizon < len(g):
            c0 = float(g["close"].iloc[pos])
            c1 = float(g["close"].iloc[pos + horizon])
            if c0 > 0 and np.isfinite(c1):
                vals.append(c1 / c0 - 1.0)
    return float(np.nanmean(vals)) if vals else np.nan


def _outcome(
    g: pd.DataFrame,
    date: pd.Timestamp,
    target: float,
    stop: float,
    horizon: int,
    cost: float,
) -> Optional[Dict[str, Any]]:
    pos = int(g["date"].searchsorted(date, side="left"))
    if pos < 0 or pos >= len(g) or g["date"].iloc[pos] != date:
        return None
    if pos + 1 >= len(g):
        return None
    entry = float(g["open"].iloc[pos + 1])
    if not np.isfinite(entry) or entry <= 0:
        return None
    if not np.isfinite(target) or target <= entry * 1.005:
        return None
    if not np.isfinite(stop) or stop >= entry:
        return None
    end = min(pos + horizon, len(g) - 1)
    exit_price = float(g["close"].iloc[end])
    reason = "hold"
    mfe = 0.0
    mae = 0.0
    for j in range(pos + 1, end + 1):
        hi = float(g["high"].iloc[j])
        lo = float(g["low"].iloc[j])
        if np.isfinite(hi):
            mfe = max(mfe, hi / entry - 1.0)
        if np.isfinite(lo):
            mae = min(mae, lo / entry - 1.0)
        # 同日先触目标/止损无法从日线判断，保守假设先止损。
        if np.isfinite(lo) and lo <= stop:
            exit_price, reason = stop, "stop"
            break
        if np.isfinite(hi) and hi >= target:
            exit_price, reason = target, "target"
            break
    gross = exit_price / entry - 1.0
    return {
        "entry": entry, "exit": exit_price, "ret": gross - cost,
        "gross_ret": gross, "reason": reason,
        "mfe": mfe, "mae": mae,
        "entry_date": str(g["date"].iloc[pos + 1].date()),
        "exit_date": str(g["date"].iloc[end].date()),
    }


def run_backtest(
    horizon: int = 10,
    step: int = 5,
    top_n: int = 10,
    cost: float = 0.002,
    min_history: int = 120,
) -> Dict[str, Any]:
    panel = load_stock_panel()
    market = load_market_panel("1.000001")
    snapshot = pd.read_parquet("data/bootstrap/snapshot.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    snap = snapshot.set_index("code")
    groups = {str(c): g.sort_values("date").reset_index(drop=True) for c, g in panel.groupby("code", sort=False)}
    dates = sorted(panel["date"].unique())
    dates = [pd.Timestamp(d) for d in dates]
    start = min_history
    end = len(dates) - horizon - 2
    if end <= start:
        return {"error": "历史长度不足", "horizon_days": horizon, "step": step}
    eval_dates = dates[start:end:step]
    bench = market.sort_values("date").reset_index(drop=True) if not market.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    date_rows: List[Dict[str, Any]] = []

    for date in eval_dates:
        sub = panel[panel["date"] <= date]
        bench_sub = bench[bench["date"] <= date] if not bench.empty else pd.DataFrame()
        date_signals: List[Dict[str, Any]] = []
        for code, g in sub.groupby("code", sort=False):
            if len(g) < min_history:
                continue
            info = snap.loc[code] if code in snap.index else {}
            cap = float(info.get("float_mcap_yi", np.nan) or np.nan)
            is_st = bool(info.get("is_st", False))
            try:
                feat = technical_features(
                    g.tail(320), benchmark=bench_sub,
                    limit_pct=float(info.get("limit_pct", 0.10) or 0.10),
                    float_shares=info.get("float_shares"),
                )
                wave = detect_wave(g.tail(280))
                first = detect_first_wave(g.tail(280))
            except Exception:
                continue
            if feat.get("bars", 0) < min_history:
                continue
            six = _six_proxy(pd.Series(feat), cap, is_st)
            base = {"code": str(code).zfill(6), "date": str(date.date()), "name": str(info.get("name", "")), **six}
            if wave.get("wave_stage_code") in ("early3", "main3") and wave.get("wave_pattern_score", 0) >= 60:
                date_signals.append({
                    **base, "strategy": "wave3_typical",
                    "score": float(wave.get("wave_profit_score", 0)),
                    "target": float(wave.get("wave_target1", np.nan)),
                    "target2": float(wave.get("wave_target2", np.nan)),
                    "stop": float(wave.get("wave_stop", np.nan)),
                    "stage": wave.get("wave_stage", ""),
                    "launch_prob": wave.get("wave_launch_prob", np.nan),
                })
            if wave.get("wave_stage_code") == "pre3" and wave.get("wave_pattern_score", 0) >= 60:
                date_signals.append({
                    **base, "strategy": "wave3_pre",
                    "score": float(wave.get("wave_profit_score", 0)),
                    "target": float(wave.get("wave_target1", np.nan)),
                    "target2": float(wave.get("wave_target2", np.nan)),
                    "stop": float(wave.get("wave_stop", np.nan)),
                    "stage": wave.get("wave_stage", ""),
                    "launch_prob": wave.get("wave_launch_prob", np.nan),
                })
            if first.get("first_wave_stage_code") in ("pre1", "pre1_break") and first.get("first_wave_pattern_score", 0) >= 55 and six["six_pass"]:
                date_signals.append({
                    **base, "strategy": "first_wave",
                    "score": float(first.get("first_wave_profit_score", 0)),
                    "target": float(first.get("first_wave_target1", np.nan)),
                    "target2": float(first.get("first_wave_target2", np.nan)),
                    "stop": float(first.get("first_wave_stop", np.nan)),
                    "stage": first.get("first_wave_stage", ""),
                    "launch_prob": first.get("first_wave_launch_prob", np.nan),
                })
            if first.get("first_wave_stage_code") in ("pre1", "pre1_break") and first.get("first_wave_pattern_score", 0) >= 55 and not six["six_pass"]:
                date_signals.append({
                    **base, "strategy": "first_wave_raw",
                    "score": float(first.get("first_wave_profit_score", 0)),
                    "target": float(first.get("first_wave_target1", np.nan)),
                    "target2": float(first.get("first_wave_target2", np.nan)),
                    "stop": float(first.get("first_wave_stop", np.nan)),
                    "stage": first.get("first_wave_stage", ""),
                    "launch_prob": first.get("first_wave_launch_prob", np.nan),
                })
        if not date_signals:
            continue
        uni = _universe_return(date, groups, horizon)
        date_rows.append({"date": str(date.date()), "signals": len(date_signals), "universe_ret": uni})
        sig_df = pd.DataFrame(date_signals)
        for strategy, sdf in sig_df.groupby("strategy"):
            selected = sdf.sort_values("score", ascending=False).head(top_n)
            for _, sig in selected.iterrows():
                out = _outcome(groups.get(str(sig["code"]), pd.DataFrame()), date, float(sig["target"]), float(sig["stop"]), horizon, cost)
                if out is None:
                    continue
                out.update({k: sig.get(k) for k in ("code", "name", "strategy", "score", "stage", "launch_prob", "six_pass", "setup_proxy", "check_proxy", "six_proxy_text")})
                out["signal_date"] = str(date.date())
                out["universe_ret"] = uni
                out["excess"] = out["ret"] - uni if np.isfinite(uni) else np.nan
                rows.append(out)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return {"error": "没有产生可执行交易", "horizon_days": horizon, "step": step}
    results: Dict[str, Any] = {}
    for strategy, g in trades.groupby("strategy"):
        ret = pd.to_numeric(g["ret"], errors="coerce").dropna()
        excess = pd.to_numeric(g["excess"], errors="coerce").dropna()
        results[str(strategy)] = {
            "trades": int(len(g)),
            "dates": int(g["signal_date"].nunique()),
            "win_rate": float((ret > 0).mean()) if len(ret) else None,
            "avg_ret": float(ret.mean()) if len(ret) else None,
            "median_ret": float(ret.median()) if len(ret) else None,
            "avg_excess": float(excess.mean()) if len(excess) else None,
            "target_rate": float((g["reason"] == "target").mean()),
            "stop_rate": float((g["reason"] == "stop").mean()),
            "avg_mfe": float(pd.to_numeric(g["mfe"], errors="coerce").mean()),
            "avg_mae": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
            "avg_hold_days": float(horizon),
        }
    return {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_days": horizon,
        "step_days": step,
        "top_n_per_date": top_n,
        "cost_one_way": cost,
        "start_date": str(eval_dates[0].date()),
        "end_date": str(eval_dates[-1].date()),
        "trades": len(trades),
        "results": results,
        "dates": date_rows,
        "notes": [
            "仅覆盖灾备包内 840 只股票，存在幸存者偏差。",
            "历史公告/减持/立案无法完整点时重建，回测未计入；实盘第一浪候选会单独核查。",
            "六维/六格为价量代理，不含历史主题/财务点时数据。",
            "日线无法判断同日目标/止损先后，统一保守假设先止损。",
            "未考虑涨跌停无法成交、停牌、滑点冲击，成本仅按 0.1% 单边估算。",
        ],
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    out = run_backtest()
    Path("data/wave_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out.get("results", out), ensure_ascii=False, indent=2))
