# -*- coding: utf-8 -*-
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spec_radar import learning
from spec_radar.altdata import AnnBundle, apply_announcements
from spec_radar.forecast import load_params, predict_forecasts
from spec_radar.framework import add_active_framework, add_potential_framework, add_trade_levels
from spec_radar.wave import detect_wave


def _base_row(**overrides):
    row = {
        "theme_fit": 0.7, "theme_tilt": 0.8, "concept_day": 0.02, "concept_20d": 0.05,
        "concept_60d": 0.10, "ind_ret20": 0.03, "float_mcap_yi": 80.0, "dd52": -0.28,
        "ret20": 0.08, "c_compression": 72.0, "c_accumulation": 68.0,
        "amount_ratio_20_120": 0.9, "cmf20": 0.08, "obv_slope20": 0.02,
        "up_down_vol_ratio": 1.4, "penalty": 0.0, "ann_red_text": "",
        "low250": 8.0, "high250": 15.0, "ma60": 10.0,
        "sector_limitup_count": 4, "consecutive_limit_up": 1.0,
        "close_signal": 12.0, "ma20": 11.5, "turnover_mean20": 0.12,
        "amplitude_today": 0.08, "open_count": 0, "lhb_net_yi": 0.2,
        "ann_good_text": "订单", "check_count": 5,
    }
    row.update(overrides)
    return row


def test_potential_six_dim_scores():
    df = pd.DataFrame([_base_row(), _base_row(theme_fit=0.0, float_mcap_yi=300, dd52=-0.02)])
    out = add_potential_framework(df)
    assert out.loc[0, "setup_score"] == 12
    assert out.loc[0, "setup_grade"] == "等买点"
    assert out.loc[0, "setup_dims"].startswith("题材2")
    assert out.loc[1, "setup_score"] < 9


def test_active_six_check_and_redline():
    df = pd.DataFrame([_base_row(), _base_row(ann_red_text="澄清/无关联")])
    out = add_active_framework(df)
    assert out.loc[0, "check_count"] == 6
    assert out.loc[0, "check_grade"] == "通过"
    assert out.loc[1, "check_safety"] == False
    assert out.loc[1, "check_grade"] == "红线剔除"


def test_forecast_predictions_are_bounded():
    params = load_params()
    if not params:
        pytest.skip("尚未训练 forecast_params")
    df = pd.DataFrame([
        {"close_signal": 10.0, "atr14": 0.03, "dd52": -0.25, "pos52": 0.5,
         "turnover_mean20": 0.05, "amount_mean20": 2e8, "dist_high60": -0.08,
         "c_compression": 70, "c_accumulation": 65, "c_base": 60, "c_neglect": 50,
         "c_momentum": 70, "c_volume": 60, "c_breakout": 50, "c_flow": 55},
    ])
    for key in ("potential", "active"):
        out = predict_forecasts(df, key, params)
        assert 0.02 <= out.iloc[0]["p_hit_10d"] <= 0.95
        assert out.iloc[0]["target_high"] >= 10.0
        assert out.iloc[0]["target_high_bull"] >= out.iloc[0]["target_high"]
        assert out.iloc[0]["launch_window"] in {"1-3", "4-10", "11-20", "20+"}


def test_wave_detection_synthetic():
    n = 110
    close = np.concatenate([
        np.linspace(10.8, 10.0, 8),
        np.linspace(10.0, 13.0, 24),
        np.linspace(13.0, 11.5, 22),
        np.linspace(11.5, 13.3, 24),
        np.linspace(13.3, 13.8, n - 78),
    ])
    df = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": close * 0.999, "high": close * 1.02,
        "low": close * 0.98, "close": close,
        "volume": np.linspace(1.0e6, 1.4e6, n),
    })
    out = detect_wave(df)
    assert out["wave_stage"] != "非典型"
    assert out["wave_pattern_score"] > 0
    assert out["wave_target2"] >= out["wave_target1"]
    assert out["wave_stop"] < out["wave_buy_low"]


def test_trade_levels_are_ordered():
    df = pd.DataFrame([{
        "close_signal": 20.0, "current_price": 21.0, "ma5": 20.2, "ma10": 19.8,
        "ma20": 19.0, "ma60": 17.5, "atr14": 0.04, "dist_low60": 0.15,
        "dist_high60": -0.10, "target_high_low": 21.5, "target_high": 22.0,
        "target_high_bull": 23.0,
    }])
    out = add_trade_levels(df, "potential")
    row = out.iloc[0]
    assert row["buy_zone_low"] < row["buy_zone_high"]
    assert row["stop_loss"] < row["best_buy_price"]
    assert row["sell_zone_low"] > row["best_buy_price"]
    assert row["sell_zone_high"] >= row["sell_zone_low"]
    assert row["risk_reward"] > 0


def test_announcement_redline_excludes_stock():
    df = pd.DataFrame([{
        "code": "000001", "score": 70.0, "confidence": 80.0, "risk_flags": [],
        "risk_flags_text": "", "amount20_yi": 2.0,
    }])
    flags = pd.DataFrame([{
        "code": "000001", "ann_checked": True, "ann_red": ["澄清/无关联"],
        "ann_red_text": "澄清/无关联", "ann_good_text": "", "ann_titles": [],
        "ann_latest": "澄清公告",
    }])
    out, hard = apply_announcements(df, AnnBundle(flags))
    assert hard == ["000001"]
    assert bool(out.iloc[0]["ann_hard_excluded"])
    assert out.iloc[0]["score"] < 70


def test_learning_roundtrip(monkeypatch):
    # 沙箱下 pytest 的系统临时目录可能被拒绝；用工作区内的隔离目录。
    tmp_path = Path(".tmp") / f"learn_test_{uuid.uuid4().hex[:8]}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(learning, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(learning, "CALIBRATION_PATH", tmp_path / "calibration.json")
    dates = pd.bdate_range("2026-01-01", periods=30)
    panel = pd.DataFrame({
        "code": ["000001"] * len(dates), "date": dates,
        "open": 10.0, "high": np.linspace(10, 13, len(dates)),
        "low": np.linspace(10, 12.5, len(dates)), "close": np.linspace(10, 12.8, len(dates)),
        "volume": 1e6, "amount": 1e7, "turnover": 1.0,
    })
    result = {
        "meta": {"signal_date": str(dates[5].date())},
        "potential": pd.DataFrame([{
            "code": "000001", "name": "测试", "rank": 1, "score": 70,
            "close_signal": 10.5, "p_hit_10d": 0.5, "p_up_10d": 0.6,
            "target_high": 12.0, "target_high_bull": 13.0, "launch_window": "4-10",
        }]),
        "active": pd.DataFrame(),
    }
    assert learning.record_predictions(result, 1, 1) == 1
    assert len(learning.load_ledger()) == 1
    status = learning.evaluate_ledger(panel)
    assert status["evaluated"] == 1
    cal = learning.calibrate()
    assert "potential" in cal["models"]
    assert len(learning.ledger_csv()) > 0
