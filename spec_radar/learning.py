# -*- coding: utf-8 -*-
"""自我学习账本与概率校准。

重要：这不是“越跑越准”的黑箱。它只做三件事：
1. 把每次 Top N 预测按信号日存档；
2. 等未来 10 日行情走完后，计算真实命中/最大有利波动/起爆日；
3. 用已结算样本做 Beta/频率校准，调整概率与目标价倍数。

Streamlit Community Cloud 的实例文件系统会在重启后重置；页面提供导出/导入账本。
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEARNING_DIR = ROOT / "data" / "learning"
LEDGER_PATH = LEARNING_DIR / "ledger.json"
CALIBRATION_PATH = LEARNING_DIR / "calibration.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_ledger() -> List[Dict[str, Any]]:
    data = _read_json(LEDGER_PATH, [])
    return data if isinstance(data, list) else []


def load_calibration() -> Dict[str, Any]:
    data = _read_json(CALIBRATION_PATH, {})
    return data if isinstance(data, dict) else {}


def record_predictions(result: Dict[str, Any], n_potential: int, n_active: int) -> int:
    """记录两个榜单的 Top N。相同信号日/模型/代码只记一次。"""
    ledger = load_ledger()
    existing = {(str(x.get("signal_date")), str(x.get("model")), str(x.get("code"))) for x in ledger}
    added = 0
    signal_date = str(result.get("meta", {}).get("signal_date", ""))
    for model_key, frame, n in (("potential", result.get("potential", pd.DataFrame()), n_potential),
                                ("active", result.get("active", pd.DataFrame()), n_active)):
        if frame is None or frame.empty:
            continue
        for _, row in frame.head(int(n)).iterrows():
            code = str(row.get("code")).zfill(6)
            key = (signal_date, model_key, code)
            if key in existing:
                continue
            ledger.append({
                "signal_date": signal_date,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_key,
                "code": code,
                "name": str(row.get("name", "")),
                "rank": int(row.get("rank", 0)),
                "score": float(row.get("score", 0)),
                "entry_close": float(row.get("close_signal", np.nan)),
                "p_hit": float(row.get("p_hit_10d", np.nan)),
                "p_up": float(row.get("p_up_10d", np.nan)),
                "hit_threshold": float(row.get("forecast_threshold", 0.08)),
                "target_high": float(row.get("target_high", np.nan)),
                "target_high_bull": float(row.get("target_high_bull", np.nan)),
                "launch_window": str(row.get("launch_window", "")),
                "evaluated": False,
            })
            added += 1
    if added:
        _write_json(LEDGER_PATH, ledger)
    return added


def _outcome_for(g: pd.DataFrame, date: pd.Timestamp, horizon: int = 10, hit_threshold: float = 0.08) -> Optional[Dict[str, Any]]:
    g = g.sort_values("date").reset_index(drop=True)
    pos = int(g["date"].searchsorted(date, side="left"))
    if pos >= len(g) or g["date"].iloc[pos] != date:
        return None
    if pos + horizon >= len(g):
        return None
    entry = float(g["close"].iloc[pos])
    future = g.iloc[pos + 1:pos + horizon + 1]
    if future.empty or entry <= 0:
        return None
    mfe = float(future["high"].max() / entry - 1)
    ret = float(g["close"].iloc[pos + horizon] / entry - 1)
    prior_high = float(g["high"].iloc[max(0, pos - 20):pos].max()) if pos > 0 else entry
    launch_days = None
    for j in range(pos + 1, min(pos + 21, len(g))):
        pct = float(g["close"].iloc[j] / g["close"].iloc[j - 1] - 1)
        if float(g["close"].iloc[j]) > prior_high or pct >= (0.197 if len(g) and str(g.get("code", pd.Series([""])).iloc[0]).startswith(("300", "301", "688", "689")) else 0.097):
            launch_days = j - pos
            break
    return {
        "mfe10": mfe,
        "ret10": ret,
        "hit_mfe": bool(mfe >= float(hit_threshold)),
        "positive": bool(ret > 0),
        "launch_days": int(launch_days) if launch_days is not None else 21,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def evaluate_ledger(panel: pd.DataFrame, horizon: int = 10) -> Dict[str, int]:
    ledger = load_ledger()
    if not ledger or panel is None or panel.empty:
        return {"evaluated": 0, "pending": 0}
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    groups = {str(code).zfill(6): g for code, g in p.groupby("code", sort=False)}
    evaluated = 0
    for item in ledger:
        if item.get("evaluated"):
            continue
        date = pd.Timestamp(str(item.get("signal_date")))
        g = groups.get(str(item.get("code")).zfill(6))
        if g is None:
            continue
        threshold = float(item.get("hit_threshold", 0.08 if item.get("model") == "potential" else 0.10))
        outcome = _outcome_for(g, date, horizon, threshold)
        if outcome is None:
            continue
        item.update(outcome)
        item["evaluated"] = True
        evaluated += 1
    if evaluated:
        _write_json(LEDGER_PATH, ledger)
    pending = sum(1 for x in ledger if not x.get("evaluated"))
    return {"evaluated": evaluated, "pending": pending, "total": len(ledger)}


def calibrate() -> Dict[str, Any]:
    ledger = load_ledger()
    evaluated = [x for x in ledger if x.get("evaluated")]
    out: Dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(ledger),
        "evaluated_records": len(evaluated),
        "models": {},
    }
    for model in ("potential", "active"):
        rows = [x for x in evaluated if x.get("model") == model]
        if not rows:
            out["models"][model] = {"ok": False, "n": 0, "note": "暂无已结算样本"}
            continue
        hit = np.array([1.0 if x.get("hit_mfe") else 0.0 for x in rows])
        p = np.array([float(x.get("p_hit", 0.5)) for x in rows])
        target = np.array([float(x.get("target_high", 0)) for x in rows])
        mfe = np.array([float(x.get("mfe10", 0)) for x in rows])
        entry = np.array([float(x.get("entry_close", 0)) for x in rows])
        n = len(rows)
        mean_p = float(np.nanmean(p)) if n else float("nan")
        factor = float(np.clip(hit.mean() / mean_p, 0.6, 1.4)) if mean_p and np.isfinite(mean_p) and mean_p > 0 else 1.0
        realized_upside = mfe
        predicted_upside = np.where(entry > 0, (target - entry) / entry, np.nan)
        valid = np.isfinite(predicted_upside) & (predicted_upside > 0.005)
        target_factor = float(np.clip(np.nanmedian(realized_upside[valid]) / np.nanmedian(predicted_upside[valid]), 0.7, 1.3)) if valid.sum() >= 10 else 1.0
        brier = float(np.mean((p - hit) ** 2)) if n else None
        # 分数分层校准：只有样本足够时，才对“高分组/低分组”做小幅修正，防止过拟合。
        score_adjustments = {}
        score_terciles = None
        if n >= 30:
            scores = np.array([float(x.get("score", np.nan)) for x in rows])
            valid_score = np.isfinite(scores)
            if valid_score.sum() >= 30:
                q1, q2 = np.quantile(scores[valid_score], [1 / 3, 2 / 3])
                score_terciles = [float(q1), float(q2)]
                overall = float(hit.mean())
                for label, lo, hi in (("low", -np.inf, q1), ("mid", q1, q2), ("high", q2, np.inf)):
                    mask = valid_score & (scores >= lo) & (scores < hi)
                    if mask.sum() >= 8:
                        bucket_hit = float(hit[mask].mean())
                        score_adjustments[label] = float(np.clip((bucket_hit - overall) * 6.0, -2.5, 2.5))
        out["models"][model] = {
            "ok": n >= 20,
            "n": n,
            "hit_rate": float(hit.mean()),
            "mean_p_hit": mean_p,
            "brier": brier,
            "probability_factor": factor if n >= 20 else 1.0,
            "target_factor": target_factor if n >= 20 else 1.0,
            "avg_mfe": float(np.nanmean(mfe)),
            "score_terciles": score_terciles,
            "score_adjustments": score_adjustments,
            "note": "样本不足，暂不应用校准" if n < 20 else "概率/目标价/分数分层已按已结算样本校准",
        }
    _write_json(CALIBRATION_PATH, out)
    return out


def apply_calibration(df: pd.DataFrame, model_key: str, calibration: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    cal = calibration if calibration is not None else load_calibration()
    m = (cal.get("models") or {}).get(model_key) or {}
    out = df.copy()
    if not m.get("ok"):
        out["learning_status"] = "冷启动/样本不足"
        return out
    factor = float(m.get("probability_factor", 1.0))
    target_factor = float(m.get("target_factor", 1.0))
    out["p_hit_10d"] = (out["p_hit_10d"] * factor).clip(0.02, 0.95)
    out["p_up_10d"] = (out["p_up_10d"] * 0.7 + out["p_hit_10d"] * 0.3).clip(0.05, 0.95)
    for col in ("target_high_low", "target_high", "target_high_bull"):
        out[col] = (out[col] * target_factor).round(2)
    out["target_upside_pct"] = out["target_high"] / out["close_signal"] - 1.0
    terciles = m.get("score_terciles")
    adjustments = m.get("score_adjustments") or {}
    if terciles and adjustments:
        def bucket(score: float) -> str:
            if score < float(terciles[0]):
                return "low"
            if score < float(terciles[1]):
                return "mid"
            return "high"
        out["learning_score_adj"] = out["score"].map(lambda s: float(adjustments.get(bucket(float(s)), 0.0)))
        out["score"] = (out["score"] + out["learning_score_adj"]).clip(0, 100)
        out = out.sort_values(["score", "confidence", "amount20_yi"], ascending=[False, False, False]).reset_index(drop=True)
        if "rank" in out:
            out["rank"] = np.arange(1, len(out) + 1)
    out["learning_status"] = f"已校准 n={int(m.get('n', 0))}"
    return out


def ledger_csv() -> bytes:
    ledger = load_ledger()
    if not ledger:
        return b""
    cols = ["signal_date", "model", "code", "name", "rank", "score", "entry_close", "p_hit", "p_up",
            "hit_threshold", "target_high", "launch_window", "evaluated", "mfe10", "ret10", "hit_mfe", "launch_days"]
    rows = [{c: x.get(c, "") for c in cols} for x in ledger]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def import_ledger_csv(data: bytes) -> int:
    if not data:
        return 0
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    existing = load_ledger()
    seen = {(str(x.get("signal_date")), str(x.get("model")), str(x.get("code"))) for x in existing}
    added = 0
    for row in reader:
        key = (str(row.get("signal_date")), str(row.get("model")), str(row.get("code")).zfill(6))
        if key in seen:
            continue
        item = dict(row)
        item["code"] = str(item.get("code")).zfill(6)
        for c in ("rank", "score", "entry_close", "p_hit", "p_up", "hit_threshold", "target_high", "mfe10", "ret10", "launch_days"):
            try:
                item[c] = float(item.get(c, "")) if item.get(c, "") not in ("", None) else np.nan
            except (TypeError, ValueError):
                item[c] = np.nan
        item["evaluated"] = str(item.get("evaluated", "")).lower() in {"true", "1", "yes"}
        existing.append(item)
        seen.add(key)
        added += 1
    if added:
        _write_json(LEDGER_PATH, existing)
    return added
