# -*- coding: utf-8 -*-
"""概率 / 目标价 / 起爆窗口预测。

定位：这是“历史相似条件下的统计倾向”，不是确定性预测。模型刻意保持简单：
- 逻辑回归估计 10 日内最大有利波动（MFE）达到阈值的概率；
- 用历史 MFE / ATR 的分位数给出目标价区间；
- 用历史上“突破前高/涨停”的首次发生日分布给出起爆窗口。

训练数据只来自仓库内置的候选股历史面板，存在幸存者偏差与样本期短的问题；
UI 必须同时展示样本量与验证指标。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bootstrap import load_market_panel, load_snapshot, load_stock_panel
from .validation import _features_at, _score_sections

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "data" / "learning" / "forecast_params.json"

MODEL_THRESHOLDS = {"potential": 0.08, "active": 0.10}
FORECAST_FEATURES = {
    "potential": ["tech_score", "atr14", "dd52", "turnover_mean20", "log_amount"],
    "active": ["tech_score", "atr14", "pos52", "turnover_mean20", "log_amount"],
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.5, iterations: int = 30) -> Tuple[np.ndarray, float]:
    """小维度 IRLS/L2 逻辑回归，返回系数（不含截距）和截距。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta = np.zeros(k + 1)
    for _ in range(iterations):
        p = np.clip(_sigmoid(Xb @ beta), 1e-5, 1 - 1e-5)
        w = p * (1 - p)
        grad = Xb.T @ (y - p)
        grad[1:] -= l2 * beta[1:]
        H = Xb.T @ (Xb * w[:, None])
        reg = np.eye(k + 1) * l2
        reg[0, 0] = 0
        try:
            step = np.linalg.solve(H + reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H + reg) @ grad
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta[1:], float(beta[0])


def _standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std


def _label_frame(panel: pd.DataFrame, date: pd.Timestamp, horizon: int = 10) -> pd.DataFrame:
    rows = []
    for code, g in panel.groupby("code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        pos = int(g["date"].searchsorted(date, side="left"))
        if pos >= len(g) or g["date"].iloc[pos] != date:
            continue
        if pos + horizon >= len(g):
            continue
        entry = float(g["close"].iloc[pos])
        future = g.iloc[pos + 1:pos + horizon + 1]
        if future.empty:
            continue
        mfe = float(future["high"].max() / entry - 1)
        ret = float(g["close"].iloc[pos + horizon] / entry - 1)
        # 起爆：收盘突破前 20 日高点，或出现涨停（按各板阈值近似 9.7%/19.7%）。
        prior_high = float(g["high"].iloc[max(0, pos - 20):pos].max()) if pos > 0 else float(g["high"].iloc[pos])
        launch_days = None
        limit_thr = 0.197 if str(code).startswith(("300", "301", "688", "689")) else 0.097
        for j in range(pos + 1, min(pos + 21, len(g))):
            pct = float(g["close"].iloc[j] / g["close"].iloc[j - 1] - 1)
            if float(g["close"].iloc[j]) > prior_high or pct >= limit_thr:
                launch_days = j - pos
                break
        rows.append({"code": str(code).zfill(6), "date": date, "fwd_mfe10": mfe, "fwd_ret10": ret,
                     "fwd_launch_days": launch_days if launch_days is not None else 21})
    return pd.DataFrame(rows)


def _tech_score(feat: pd.DataFrame, model_key: str) -> pd.Series:
    if model_key == "potential":
        return (
            0.38 * feat.get("v_potential_score", pd.Series(50, index=feat.index))
        )
    return 0.35 * feat.get("v_active_score", pd.Series(50, index=feat.index))


def _prepare_dataset(step: int = 10) -> Dict[str, pd.DataFrame]:
    panel = load_stock_panel()
    market = load_market_panel("1.000001")
    snapshot = load_snapshot()
    if panel.empty or market.empty:
        raise RuntimeError("bootstrap 数据不足，无法训练预测模型")
    panel["date"] = pd.to_datetime(panel["date"])
    market["date"] = pd.to_datetime(market["date"])
    dates = sorted(panel["date"].unique())
    if len(dates) < 220:
        raise RuntimeError("历史面板不足 220 个交易日")
    eval_dates = [pd.Timestamp(d) for d in dates[180:len(dates) - 21:step]]
    frames = []
    for date in eval_dates:
        feat = _features_at(panel, market, snapshot, date)
        if feat.empty:
            continue
        feat = _score_sections(feat)
        labels = _label_frame(panel, date, horizon=10)
        if labels.empty:
            continue
        f = feat.reset_index(drop=True).merge(labels, on=["code", "date"], how="inner")
        if f.empty:
            continue
        f["mfe_ratio"] = f["fwd_mfe10"] / np.maximum(f["atr14"] * math.sqrt(10), 0.008)
        f["hit_mfe"] = (f["fwd_mfe10"] >= 0.08).astype(float)  # 两条线会按模型阈值重算
        f["positive"] = (f["fwd_ret10"] > 0).astype(float)
        f["log_amount"] = np.log1p(f["amount_mean20"].clip(lower=0))
        frames.append(f)
    if not frames:
        raise RuntimeError("没有生成任何训练截面")
    data = pd.concat(frames, ignore_index=True)
    return {"data": data, "snapshot": snapshot}


def train_forecast(step: int = 10) -> Dict[str, Any]:
    prepared = _prepare_dataset(step=step)
    data = prepared["data"]
    result: Dict[str, Any] = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": "2026.10-r1",
        "horizon_days": 10,
        "thresholds": MODEL_THRESHOLDS,
        "caveat": "候选股历史面板训练，存在幸存者偏差；概率是统计倾向，不是保证。",
        "models": {},
    }
    for model_key in ("potential", "active"):
        if model_key == "potential":
            d = data[
                (data["bars"] >= 170) & (data["ret60"].fillna(0) <= 0.60)
                & (data["ret20"].fillna(0) <= 0.32) & (data["turnover_mean20"].fillna(0) <= 0.14)
                & (data["amount_mean20"].fillna(0) >= 0.45e8)
            ].copy()
            d["tech_score"] = d["v_potential_score"]
        else:
            d = data[
                (data["ret20"].fillna(-1) >= 0) & (data["close_signal"] >= data["ma20"])
                & (data["turnover_mean20"].fillna(0) >= 0.015)
                & (data["amount_mean20"].fillna(0) >= 1e8)
            ].copy()
            d["tech_score"] = d["v_active_score"]
        threshold = MODEL_THRESHOLDS[model_key]
        d["target_hit"] = (d["fwd_mfe10"] >= threshold).astype(float)
        d = d.replace([np.inf, -np.inf], np.nan)
        feat_cols = FORECAST_FEATURES[model_key]
        d = d.dropna(subset=feat_cols + ["target_hit", "positive", "fwd_mfe10", "date"])
        dates = sorted(d["date"].unique())
        if len(dates) < 4 or len(d) < 150:
            result["models"][model_key] = {"ok": False, "reason": "样本不足", "rows": int(len(d))}
            continue
        split = max(1, int(len(dates) * 0.7))
        train_dates = set(dates[:split])
        train = d[d["date"].isin(train_dates)]
        test = d[~d["date"].isin(train_dates)]
        X_train = train[feat_cols].to_numpy(dtype=float)
        X_test = test[feat_cols].to_numpy(dtype=float)
        X_train_s, X_test_s, mean, std = _standardize(X_train, X_test)
        coef_hit, intercept_hit = _fit_logistic(X_train_s, train["target_hit"].to_numpy())
        coef_up, intercept_up = _fit_logistic(X_train_s, train["positive"].to_numpy())
        p_test_hit = _sigmoid(intercept_hit + X_test_s @ coef_hit)
        p_test_up = _sigmoid(intercept_up + X_test_s @ coef_up)
        # 最终部署参数用全样本重拟合
        X_all = d[feat_cols].to_numpy(dtype=float)
        X_all_s, _, mean_all, std_all = _standardize(X_all, X_all)
        coef_hit_all, intercept_hit_all = _fit_logistic(X_all_s, d["target_hit"].to_numpy())
        coef_up_all, intercept_up_all = _fit_logistic(X_all_s, d["positive"].to_numpy())
        terciles = np.quantile(d["tech_score"], [1 / 3, 2 / 3])
        buckets = {}
        for label, lo, hi in (("low", -np.inf, terciles[0]), ("mid", terciles[0], terciles[1]), ("high", terciles[1], np.inf)):
            sub = d[(d["tech_score"] >= lo) & (d["tech_score"] < hi)]
            if len(sub) < 30:
                sub = d
            ratios = sub["mfe_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            q = ratios.quantile([0.25, 0.5, 0.75]).to_dict() if len(ratios) else {0.25: 1.0, 0.5: 1.5, 0.75: 2.5}
            launch = sub["fwd_launch_days"].fillna(21)
            buckets[label] = {
                "n": int(len(sub)),
                "mfe_ratio_q25": float(q.get(0.25, 1.0)),
                "mfe_ratio_q50": float(q.get(0.5, 1.5)),
                "mfe_ratio_q75": float(q.get(0.75, 2.5)),
                "launch_probs": {
                    "1-3": float((launch <= 3).mean()),
                    "4-10": float(((launch > 3) & (launch <= 10)).mean()),
                    "11-20": float(((launch > 10) & (launch <= 20)).mean()),
                    "20+": float((launch > 20).mean()),
                },
                "launch_median": float(launch.median()),
            }
        result["models"][model_key] = {
            "ok": True,
            "rows": int(len(d)),
            "dates": int(len(dates)),
            "train_dates": int(len(train_dates)),
            "test_rows": int(len(test)),
            "positive_rate": float(d["target_hit"].mean()),
            "up_rate": float(d["positive"].mean()),
            "auc_hit": _auc(test["target_hit"].to_numpy(), p_test_hit),
            "auc_up": _auc(test["positive"].to_numpy(), p_test_up),
            "brier_hit": float(np.mean((p_test_hit - test["target_hit"].to_numpy()) ** 2)) if len(test) else None,
            "threshold": threshold,
            "features": feat_cols,
            "mean": mean_all.tolist(),
            "std": std_all.tolist(),
            "coef_hit": coef_hit_all.tolist(),
            "intercept_hit": intercept_hit_all,
            "coef_up": coef_up_all.tolist(),
            "intercept_up": intercept_up_all,
            "terciles": terciles.tolist(),
            "buckets": buckets,
        }
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_params() -> Dict[str, Any]:
    if not PARAMS_PATH.exists():
        return {}
    try:
        return json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tech_score_from_components(df: pd.DataFrame, model_key: str) -> pd.Series:
    if model_key == "potential":
        return (
            0.38 * df.get("c_compression", 50)
            + 0.34 * df.get("c_accumulation", 50)
            + 0.18 * df.get("c_base", 50)
            + 0.10 * df.get("c_neglect", 50)
        )
    return (
        0.35 * df.get("c_momentum", 50)
        + 0.30 * df.get("c_volume", 50)
        + 0.20 * df.get("c_breakout", 50)
        + 0.15 * df.get("c_flow", 50)
    )


def _bucket_for(score: float, terciles: List[float]) -> str:
    if score < terciles[0]:
        return "low"
    if score < terciles[1]:
        return "mid"
    return "high"


def predict_forecasts(df: pd.DataFrame, model_key: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """给已评分的榜单添加概率、目标价区间、起爆窗口。"""
    out = df.copy()
    if out.empty:
        return out
    params = params if params is not None else load_params()
    model = (params.get("models") or {}).get(model_key) or {}
    tech = _tech_score_from_components(out, model_key).clip(0, 100)
    atr = out["atr14"].fillna(0.025).clip(lower=0.005, upper=0.20)
    threshold = float(model.get("threshold", MODEL_THRESHOLDS[model_key]))
    if model.get("ok"):
        X = pd.DataFrame({
            "tech_score": tech,
            "atr14": atr,
            "dd52": out["dd52"].fillna(-0.2),
            "pos52": out["pos52"].fillna(0.5),
            "turnover_mean20": out["turnover_mean20"].fillna(0.03),
            "log_amount": np.log1p(out["amount_mean20"].clip(lower=0)),
        })[model["features"]].to_numpy(dtype=float)
        mean = np.array(model["mean"], dtype=float)
        std = np.array(model["std"], dtype=float)
        Xs = (X - mean) / np.where(std < 1e-9, 1.0, std)
        p_hit = _sigmoid(float(model["intercept_hit"]) + Xs @ np.array(model["coef_hit"], dtype=float))
        p_up = _sigmoid(float(model["intercept_up"]) + Xs @ np.array(model["coef_up"], dtype=float))
        terciles = model.get("terciles", [33, 66])
        buckets = model.get("buckets", {})
    else:
        # 冷启动：保守的经验值，绝不装作已校准。
        p_hit = (0.30 if model_key == "potential" else 0.25) + 0.002 * (tech - 50)
        p_up = 0.45 + 0.001 * (tech - 50)
        terciles = [33, 66]
        buckets = {}

    labels, low, base, bull, expected_days, window_probs = [], [], [], [], [], []
    for i, (_, row) in enumerate(out.iterrows()):
        bucket = _bucket_for(float(tech.iloc[i]), terciles)
        b = buckets.get(bucket) or buckets.get("mid") or {}
        ratio_q25 = float(b.get("mfe_ratio_q25", 0.8))
        ratio_q50 = float(b.get("mfe_ratio_q50", 1.2))
        ratio_q75 = float(b.get("mfe_ratio_q75", 2.0))
        vol_pct = float(atr.iloc[i]) * math.sqrt(10)
        close = float(row["close_signal"])
        up_low = max(0.0, ratio_q25 * vol_pct)
        up_mid = max(0.0, ratio_q50 * vol_pct)
        up_high = max(up_mid, ratio_q75 * vol_pct)
        # 目标价至少考虑一部分“回到 60 日高点”的空间；但不做无限外推。
        room_to_high = max(0.0, -float(row.get("dist_high60"))) if pd.notna(row.get("dist_high60")) else 0.0
        target_mid = max(up_mid, 0.5 * room_to_high)
        target_mid = float(np.clip(target_mid, 0.005, 0.35))
        target_bull = float(np.clip(max(up_high, target_mid), 0.01, 0.50))
        low.append(close * (1 + up_low))
        base.append(close * (1 + target_mid))
        bull.append(close * (1 + target_bull))
        probs = b.get("launch_probs") or {"1-3": 0.18, "4-10": 0.32, "11-20": 0.20, "20+": 0.30}
        label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else "10-20"
        labels.append(label)
        window_probs.append(
            f"1-3日 {probs.get('1-3', 0):.0%} / 4-10日 {probs.get('4-10', 0):.0%} / "
            f"11-20日 {probs.get('11-20', 0):.0%} / 20日+ {probs.get('20+', 0):.0%}"
        )
        expected_days.append(float(b.get("launch_median", 12)))
    out["p_hit_10d"] = np.clip(p_hit, 0.02, 0.95)
    out["p_up_10d"] = np.clip(p_up, 0.05, 0.95)
    out["forecast_threshold"] = threshold
    out["target_high_low"] = np.round(low, 2)
    out["target_high"] = np.round(base, 2)
    out["target_high_bull"] = np.round(bull, 2)
    out["target_upside_pct"] = (out["target_high"] / out["close_signal"] - 1.0)
    out["launch_window"] = labels
    out["launch_window_probs"] = window_probs
    out["expected_launch_days"] = np.round(expected_days, 0)
    out["forecast_source"] = "calibrated" if model.get("ok") else "cold_start"
    auc = model.get("auc_hit")
    out["forecast_quality"] = (
        "较高" if isinstance(auc, (int, float)) and np.isfinite(auc) and auc >= 0.70
        else "中等" if isinstance(auc, (int, float)) and np.isfinite(auc) and auc >= 0.60
        else "低"
    )
    return out
