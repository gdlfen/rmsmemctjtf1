# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from spec_radar.config import ScanSettings, load_model_config
from spec_radar.scoring import base_eligibility, score_model


def make_features(n=60, seed=42):
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    df = pd.DataFrame(index=codes)
    df["code"] = codes
    df["bars"] = 300
    df["close_signal"] = rng.uniform(5, 60, n)
    df["ma20"] = df["close_signal"] * rng.uniform(0.92, 1.05, n)
    df["ma60"] = df["close_signal"] * rng.uniform(0.88, 1.08, n)
    df["ret5"] = rng.normal(0.01, 0.05, n)
    df["ret20"] = rng.normal(0.03, 0.10, n)
    df["ret60"] = rng.normal(0.08, 0.18, n)
    df["ret250"] = rng.normal(0.10, 0.30, n)
    df["rs20"] = rng.normal(0, 0.08, n)
    df["rs60"] = rng.normal(0, 0.12, n)
    df["atr14"] = rng.uniform(0.01, 0.06, n)
    df["range60"] = rng.uniform(0.08, 0.45, n)
    df["bb_width"] = rng.uniform(0.05, 0.35, n)
    df["vol_ratio_20_120"] = rng.uniform(0.5, 1.8, n)
    df["vol20"] = rng.uniform(0.15, 0.8, n)
    df["vcp_ratio"] = rng.uniform(0.3, 1.2, n)
    df["amount_ratio_20_120"] = rng.uniform(0.3, 2.0, n)
    df["amount_ratio_5_20"] = rng.uniform(0.5, 2.0, n)
    df["amount_mean5"] = rng.uniform(2e7, 2e9, n)
    df["avg_amount20_yi"] = rng.uniform(0.5, 30, n)
    df["turnover_mean20"] = rng.uniform(0.002, 0.15, n)
    df["turnover_mean5"] = df["turnover_mean20"] * rng.uniform(0.7, 1.8, n)
    df["volume_expansion"] = df["amount_ratio_5_20"]
    df["obv_slope20"] = rng.normal(0, 0.02, n)
    df["obv_divergence"] = rng.normal(0, 0.02, n)
    df["cmf20"] = rng.normal(0, 0.2, n)
    df["up_down_vol_ratio"] = rng.uniform(0.2, 3.0, n)
    df["dd52"] = -rng.uniform(0.02, 0.55, n)
    df["dist_high60"] = -rng.uniform(0.0, 0.30, n)
    df["dist_low60"] = rng.uniform(0.05, 0.5, n)
    df["pos52"] = rng.uniform(0.1, 1.0, n)
    df["days_since_high"] = rng.integers(0, 250, n)
    df["ma20_slope"] = rng.normal(0, 0.03, n)
    df["macd_hist"] = rng.normal(0, 0.01, n)
    df["maxdd60"] = -rng.uniform(0.03, 0.40, n)
    df["above_ma120"] = rng.integers(0, 2, n)
    df["above_ma250"] = rng.integers(0, 2, n)
    df["trend_align"] = rng.integers(0, 6, n)
    df["breakout20"] = rng.integers(0, 2, n)
    df["breakout60"] = (rng.random(n) < 0.25).astype(int)
    df["near_high60"] = rng.uniform(0.7, 1.05, n)
    df["rsi14"] = rng.uniform(20, 90, n)
    df["dist_ma20"] = rng.normal(0, 0.10, n)
    df["limit_up20"] = rng.integers(0, 3, n)
    df["limit_down20"] = rng.integers(0, 2, n)
    df["big_down5"] = rng.integers(0, 3, n)
    df["beta60"] = rng.uniform(0.6, 1.8, n)
    df["limit_pct"] = 0.10
    df["float_mcap_yi"] = rng.uniform(15, 700, n)
    df["theme_fit"] = rng.uniform(0, 1, n)
    df["theme_tilt"] = rng.uniform(0, 0.95, n)
    df["theme_names"] = [("AI 算力",) for _ in range(n)]
    df["theme_id"] = "ai"
    df["theme_proxy_ret20"] = rng.normal(0.02, 0.12, n)
    df["ind_ret20"] = rng.normal(0.02, 0.08, n)
    df["ind_ret60"] = rng.normal(0.04, 0.15, n)
    df["ind_vol20"] = rng.uniform(0.1, 0.5, n)
    df["ind_breadth"] = rng.uniform(0.2, 0.8, n)
    df["ind_day"] = rng.normal(0, 0.02, n)
    df["concept_day"] = rng.normal(0.01, 0.025, n)
    df["concept_20d"] = rng.normal(0.03, 0.10, n)
    df["concept_60d"] = rng.normal(0.06, 0.18, n)
    df["concept_breadth"] = 0.5
    df["concept_leader_pct"] = 0
    df["concept_count"] = rng.integers(1, 8, n)
    df["profit_yoy"] = rng.normal(20, 80, n)
    df["rev_yoy"] = rng.normal(10, 30, n)
    df["roe"] = rng.normal(8, 8, n)
    df["gross_margin"] = rng.uniform(5, 60, n)
    df["pe_ttm"] = rng.uniform(5, 150, n)
    df["net_profit"] = rng.uniform(-1e8, 2e9, n)
    # 保证潜在模型 20 日/60 日/换手约束有足够样本
    df.loc[df.index[: n // 2], "ret60"] = rng.uniform(-0.2, 0.35, n // 2)
    df.loc[df.index[: n // 2], "ret20"] = rng.uniform(-0.12, 0.20, n // 2)
    df.loc[df.index[: n // 2], "turnover_mean20"] = rng.uniform(0.003, 0.09, n // 2)
    df.loc[df.index[: n // 2], "limit_up20"] = 0
    # 保证活跃模型有足够样本
    df.loc[df.index[n // 2:], "ret20"] = rng.uniform(0.0, 0.35, n - n // 2)
    df.loc[df.index[n // 2:], "turnover_mean20"] = rng.uniform(0.02, 0.25, n - n // 2)
    df.loc[df.index[n // 2:], "close_signal"] = df["ma20"].values[n // 2:] * rng.uniform(1.0, 1.15, n - n // 2)
    return df


def make_snapshot(features):
    n = len(features)
    return pd.DataFrame({
        "code": features.index,
        "name": [f"测试{i}" for i in range(n)],
        "industry": ["半导体"] * n,
        "board": ["主板"] * n,
        "price": features["close_signal"].values,
        "float_mcap_yi": features["float_mcap_yi"].values,
        "pe_ttm": features["pe_ttm"].values,
        "pb": 3.0,
        "net_profit": features["net_profit"].values,
        "roe": features["roe"].values,
        "profit_yoy": features["profit_yoy"].values,
        "rev_yoy": features["rev_yoy"].values,
        "gross_margin": features["gross_margin"].values,
        "main_inflow_yi": 0.1,
        "concepts": [["CPO概念", "算力概念"]] * n,
        "region": ["广东"] * n,
        "list_date": pd.Timestamp("2020-01-01"),
        "is_st": False,
    })


def macro_neutral():
    return {
        "risk_appetite": 50, "label": "震荡", "theme_returns": {"ai": 0.0},
        "gold20": 0.0, "nasdaq20": 0.0, "hstech20": 0.0, "bond20": 0.0,
    }


def test_score_models_are_bounded_and_sorted():
    cfg = load_model_config()
    features = make_features(80)
    snapshot = make_snapshot(features)
    settings = ScanSettings(n_potential=10, n_active=10, scan_potential=100, scan_active=100)
    for key in ("potential", "active"):
        out = score_model(features, snapshot, cfg, key, settings, macro_neutral(), pd.Timestamp("2026-10-16"))
        assert not out.empty, key
        assert out["score"].between(0, 100).all()
        assert out["rank"].is_monotonic_increasing
        assert out["score"].is_monotonic_decreasing
        assert out["risk_flags"].map(lambda x: isinstance(x, list)).all()
        comp_cols = [c for c in out.columns if c.startswith("c_")]
        assert len(comp_cols) == 8
        assert out[comp_cols].notna().any().all()


def test_potential_rejects_extended_and_illiquid():
    cfg = load_model_config()
    features = make_features(30)
    snapshot = make_snapshot(features)
    settings = ScanSettings(n_potential=10, n_active=10, scan_potential=100, scan_active=100)
    features.loc[features.index[0], "ret60"] = 0.90
    features.loc[features.index[1], "turnover_mean20"] = 0.30
    features.loc[features.index[2], "avg_amount20_yi"] = 0.10
    out = score_model(features, snapshot, cfg, "potential", settings, macro_neutral(), pd.Timestamp("2026-10-16"))
    assert features.index[0] not in out.index
    assert features.index[1] not in out.index
    assert features.index[2] not in out.index


def test_base_eligibility_filters_st_and_bse():
    snap = pd.DataFrame({
        "code": ["000001", "830001", "600001", "600002"],
        "name": ["正常", "北交", "ST测试", "正常2"],
        "is_st": [False, False, True, False],
        "is_suspended": [False, False, False, True],
        "board": ["主板", "北交所", "主板", "主板"],
        "price": [10.0, 10.0, 10.0, 10.0],
        "age_days": [1000, 1000, 1000, 1000],
        "float_mcap_yi": [100, 100, 100, 100],
        "amount": [1e8, 1e8, 1e8, 1e8],
        "list_date": [pd.Timestamp("2020-01-01")] * 4,
    })
    settings = ScanSettings(boards=("主板", "创业板", "科创板"))
    out = base_eligibility(snap, settings, pd.Timestamp("2026-10-16"))
    assert list(out["code"]) == ["000001"]
