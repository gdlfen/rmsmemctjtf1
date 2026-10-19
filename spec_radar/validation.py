# -*- coding: utf-8 -*-
"""价量核心子模型的滚动验证。

这是“因子有效性体检”，不是完整模型的实盘准确率：
- 只验证价量/筹码/动量分项，不包含主题映射、宏观环境和最新财务（这些缺少历史点时数据）；
- 股票池来自当前候选集，存在幸存者偏差；
- 样本期有限，结果只用于判断因子方向是否稳定，不能外推为收益承诺。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .bootstrap import load_stock_panel, load_market_panel, load_meta
from .indicators import technical_features

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = ROOT / "data" / "validation.json"


def _rank(s: pd.Series, high: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return pd.Series(50.0, index=s.index)
    r = s.rank(pct=True)
    if not high:
        r = 1 - r
    return r * 100


def _features_at(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    snapshot: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    sub = panel[panel["date"] <= date]
    if sub.empty:
        return pd.DataFrame()
    meta = snapshot.set_index("code")[["limit_pct", "float_shares"]].to_dict("index")
    bench = benchmark[benchmark["date"] <= date] if not benchmark.empty else pd.DataFrame()
    rows = []
    for code, group in sub.groupby("code", sort=False):
        info = meta.get(str(code), {})
        if len(group) < 90:
            continue
        try:
            feat = technical_features(
                group.tail(320), benchmark=bench,
                limit_pct=float(info.get("limit_pct", 0.10) or 0.10),
                float_shares=info.get("float_shares"),
            )
        except Exception:
            continue
        if feat.get("bars", 0) < 90:
            continue
        feat["code"] = str(code).zfill(6)
        feat["date"] = date
        rows.append(feat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("code", drop=False)


def _score_sections(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    # 潜在：压缩 + 吸筹 + 位置 + 低谷关注
    compression = (
        0.25 * _rank(f["atr14"], False)
        + 0.20 * _rank(f["range60"], False)
        + 0.20 * _rank(f["bb_width"], False)
        + 0.15 * _rank(f["vol_ratio_20_120"], False)
        + 0.10 * _rank(f["vcp_ratio"], False)
        + 0.10 * _rank(f["amount_ratio_20_120"], False)
    )
    accumulation = (
        0.35 * _rank(f["obv_slope20"], True)
        + 0.30 * _rank(f["cmf20"], True)
        + 0.20 * _rank(f["obv_divergence"], True)
        + 0.15 * _rank(f["up_down_vol_ratio"], True)
    )
    dd = f["dd52"].fillna(-0.30)
    dd_score = (1 - (dd + 0.22).abs() / 0.38).clip(0, 1) * 100
    support = 20 + 40 * f["above_ma120"].fillna(0) + 40 * f["above_ma250"].fillna(0)
    base_len = np.clip(f["days_since_high"].fillna(0) / 120, 0, 1) * 100
    slope = np.clip((f["ma20_slope"].fillna(0) + 0.03) / 0.06, 0, 1) * 100
    base = (
        0.40 * dd_score + 0.25 * support + 0.20 * base_len
        + 0.15 * slope
    )
    neglect = (
        0.40 * _rank(f["turnover_mean20"], False)
        + 0.35 * _rank(f["ret20"].abs(), False)
        + 0.25 * _rank(f["limit_up20"], False)
    )
    f["v_potential_score"] = (
        0.38 * compression + 0.34 * accumulation + 0.18 * base + 0.10 * neglect
    )
    # 活跃：动量 + 量能 + 突破 + 资金
    momentum = (
        0.25 * _rank(f["ret5"], True)
        + 0.35 * _rank(f["ret20"], True)
        + 0.20 * _rank(f["ret60"], True)
        + 0.20 * _rank(f["ma20_slope"], True)
    )
    volume = (
        0.35 * _rank(np.log1p(f["amount_mean5"].clip(lower=0)), True)
        + 0.30 * _rank(f["turnover_mean5"], True)
        + 0.35 * _rank(f["volume_expansion"], True)
    )
    near60 = np.clip((1 + f["dist_high60"].fillna(-0.25) - 0.75) / 0.25, 0, 1) * 100
    breakout_flag = np.where(f["breakout60"].fillna(0) > 0, 100.0, np.where(f["breakout20"].fillna(0) > 0, 80.0, 25.0))
    breakout = (
        0.35 * near60 + 0.25 * f["pos52"].fillna(0.5) * 100
        + 0.25 * f["trend_align"].fillna(0) / 5 * 100
        + 0.15 * breakout_flag
    )
    flow = (
        0.40 * _rank(f["obv_slope20"], True)
        + 0.35 * _rank(f["cmf20"], True)
        + 0.25 * _rank(f["up_down_vol_ratio"], True)
    )
    overheat = (
        7 * (f["rsi14"] > 82).astype(float)
        + 7 * (f["dist_ma20"] > 0.20).astype(float)
        + 6 * (f["ret5"] > 0.25).astype(float)
        + 5 * (f["ret20"] > 0.40).astype(float)
        + 5 * (f["ret60"] > 1.0).astype(float)
        + 4 * (f["turnover_mean20"] > 0.20).astype(float)
    ).clip(0, 30)
    f["v_active_score"] = 0.35 * momentum + 0.30 * volume + 0.20 * breakout + 0.15 * flow - overheat
    return f


def _metrics(df: pd.DataFrame, score_col: str, fwd_col: str, top_n: int) -> Dict[str, Any]:
    rows = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=[score_col, fwd_col])
        if len(g) < max(30, top_n * 2):
            continue
        # 不依赖 scipy：对两列先做平均秩，再计算 Pearson 相关 = Spearman 相关。
        ic = g[score_col].rank().corr(g[fwd_col].rank())
        top = g.nlargest(min(top_n, len(g)), score_col)
        bottom = g.nsmallest(min(top_n, len(g)), score_col)
        universe = g[fwd_col].mean()
        rows.append({
            "date": str(pd.Timestamp(date).date()),
            "n": int(len(g)),
            "rank_ic": float(ic) if pd.notna(ic) else np.nan,
            "top_return": float(top[fwd_col].mean()),
            "bottom_return": float(bottom[fwd_col].mean()),
            "universe_return": float(universe),
            "top_excess": float(top[fwd_col].mean() - universe),
            "hit_rate": float((top[fwd_col] > 0).mean()),
            "beat_rate": float(top[fwd_col].mean() > universe),
        })
    if not rows:
        return {"ok": False, "note": "样本不足"}
    detail = pd.DataFrame(rows)
    ic_mean = float(detail["rank_ic"].mean())
    ic_std = float(detail["rank_ic"].std(ddof=1)) if len(detail) > 1 else np.nan
    out = {
        "ok": True,
        "rebalances": int(len(detail)),
        "avg_universe": float(detail["n"].mean()),
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "icir": float(ic_mean / ic_std) if ic_std and np.isfinite(ic_std) and ic_std > 0 else None,
        "positive_ic_rate": float((detail["rank_ic"] > 0).mean()),
        "top_avg_return": float(detail["top_return"].mean()),
        "universe_avg_return": float(detail["universe_return"].mean()),
        "top_avg_excess": float(detail["top_excess"].mean()),
        "top_hit_rate": float(detail["hit_rate"].mean()),
        "beat_universe_rate": float(detail["beat_rate"].mean()),
        "period_start": str(detail["date"].iloc[0]),
        "period_end": str(detail["date"].iloc[-1]),
        "detail": detail.to_dict("records"),
    }
    return out


def run_validation(horizon: int = 10, step: int = 20) -> Dict[str, Any]:
    panel = load_stock_panel()
    market = load_market_panel("1.000001")
    snapshot = pd.read_parquet(ROOT / "data" / "bootstrap" / "snapshot.parquet")
    snapshot["code"] = snapshot["code"].astype(str).str.zfill(6)
    if panel.empty or market.empty:
        return {"ok": False, "error": "bootstrap 数据不足"}
    panel["date"] = pd.to_datetime(panel["date"])
    market["date"] = pd.to_datetime(market["date"])
    dates = sorted(panel["date"].unique())
    if len(dates) < 200:
        return {"ok": False, "error": "历史长度不足"}
    # 从有约 170 根历史的位置开始，留出 forward 窗口。
    start_idx = 175
    end_idx = len(dates) - horizon - 1
    eval_dates = [pd.Timestamp(d) for d in dates[start_idx:end_idx:step]]
    groups = {
        str(code): g.sort_values("date").reset_index(drop=True)
        for code, g in panel.groupby("code", sort=False)
    }
    all_rows = []
    for i, date in enumerate(eval_dates):
        feat = _features_at(panel, market, snapshot, date)
        if feat.empty:
            continue
        feat = _score_sections(feat)
        # 未来 horizon 日收益（预先按股票分组，避免每个截面重复扫描全表）
        fwd = {}
        for code in feat.index:
            g = groups.get(str(code))
            if g is None or g.empty:
                continue
            pos = int(g["date"].searchsorted(date, side="left"))
            if pos + horizon < len(g) and g["date"].iloc[pos] == date:
                fwd[code] = float(g["close"].iloc[pos + horizon] / g["close"].iloc[pos] - 1)
        feat["fwd"] = pd.Series(fwd).reindex(feat.index)
        feat = feat.dropna(subset=["fwd"])
        # 潜在域
        p = feat[
            (feat["bars"] >= 170) & (feat["ret60"].fillna(0) <= 0.60)
            & (feat["ret20"].fillna(0) <= 0.32) & (feat["turnover_mean20"].fillna(0) <= 0.14)
            & (feat["amount_mean20"].fillna(0) >= 0.45e8)
        ].copy()
        if len(p) >= 40:
            p["date"] = date
            all_rows.append(p[["date", "v_potential_score", "fwd"]].assign(model="potential"))
        # 活跃域
        a = feat[
            (feat["ret20"].fillna(-1) >= 0) & (feat["close_signal"] >= feat["ma20"])
            & (feat["turnover_mean20"].fillna(0) >= 0.015)
            & (feat["amount_mean20"].fillna(0) >= 1e8)
        ].copy()
        if len(a) >= 25:
            a["date"] = date
            all_rows.append(a[["date", "v_active_score", "fwd"]].assign(model="active"))
        print(f"{date.date()}: potential={len(p)} active={len(a)}")
    if not all_rows:
        return {"ok": False, "error": "没有有效评估样本"}
    all_df = pd.concat(all_rows, ignore_index=True)
    pot = all_df[all_df["model"] == "potential"]
    act = all_df[all_df["model"] == "active"]
    result = {
        "ok": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": str(pd.Timestamp(panel["date"].max()).date()),
        "horizon_days": horizon,
        "rebalance_step": step,
        "universe": "当前扫描候选池（存在幸存者偏差）",
        "method": "仅价量核心分项；月度滚动截面 Spearman IC 与 TopN 前瞻收益",
        "potential": _metrics(pot, "v_potential_score", "fwd", 20),
        "active": _metrics(act, "v_active_score", "fwd", 10),
        "notes": [
            "未纳入主题、宏观、最新财务，因此不是完整模型的准确率。",
            "候选池来自当前数据，历史已退市/未入选股票不在样本中。",
            "TopN 为每个调仓日按子模型分数取前 N，等权；收益未扣交易成本。",
            "样本期短、市场风格单一，结果只用于因子方向体检，不构成收益承诺。",
        ],
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_validation(), ensure_ascii=False, indent=2, default=str)[:5000])
