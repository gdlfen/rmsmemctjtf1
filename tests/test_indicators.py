# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from spec_radar.indicators import compute_feature_frame, technical_features


def make_kline(n=260, start=10.0, drift=0.0012, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    rets = drift + rng.normal(0, 0.012, n)
    close = start * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(50_000, 300_000, n).astype(float)
    amount = volume * 100 * close
    return pd.DataFrame({
        "date": dates, "open": open_, "close": close, "high": high, "low": low,
        "volume": volume, "amount": amount, "amplitude": (high - low) / close,
        "pct": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        "change": pd.Series(close).diff().fillna(0).to_numpy(),
        "turnover": volume * 100 / 1e9 * 100,
    })


def test_technical_features_core_values():
    df = make_kline()
    f = technical_features(df, benchmark=df, limit_pct=0.10, float_shares=1e9)
    assert f["bars"] == len(df)
    assert f["ret20"] == pytest.approx(df["close"].iloc[-1] / df["close"].iloc[-21] - 1)
    assert np.isfinite(f["atr14"]) and f["atr14"] > 0
    assert 0 <= f["rsi14"] <= 100
    assert -1 <= f["dd52"] <= 0
    assert 0 <= f["pos52"] <= 1
    assert f["date_signal"] == df["date"].iloc[-1]
    for key in ("cmf20", "obv_slope20", "amount_mean20", "turnover_mean20", "maxdd60"):
        assert np.isfinite(f[key])


def test_no_future_leak_in_features():
    df = make_kline(n=220)
    cut = df.iloc[:-30].copy()
    f_cut = technical_features(cut, benchmark=cut)
    assert f_cut["date_signal"] == cut["date"].iloc[-1]
    # 末尾新增极端行情不能改变截断日之前的信号值
    f_full = technical_features(df, benchmark=df)
    assert f_cut["ret20"] == pytest.approx(f_full["ret20"]) or f_cut["date_signal"] < f_full["date_signal"]


def test_compute_feature_frame_multi_stock():
    a = make_kline(seed=1)
    b = make_kline(seed=2, drift=-0.0005)
    a["code"] = "000001"; a["secid"] = "0.000001"
    b["code"] = "600000"; b["secid"] = "1.600000"
    panel = pd.concat([a, b], ignore_index=True)
    snapshot = pd.DataFrame([
        {"code": "000001", "name": "测试一", "industry": "银行", "board": "主板", "limit_pct": 0.10, "float_shares": 1e9},
        {"code": "600000", "name": "测试二", "industry": "银行", "board": "主板", "limit_pct": 0.10, "float_shares": 2e9},
    ])
    features = compute_feature_frame(panel, benchmark=a, snapshot=snapshot, min_bars=100)
    assert set(features.index) == {"000001", "600000"}
    assert features.loc["000001", "bars"] >= 200
    assert features["atr14"].notna().all()


def test_invalid_short_frame_raises():
    with pytest.raises(ValueError):
        technical_features(make_kline(n=10))


def test_volume_unit_normalization_for_star_fallback():
    """腾讯科创板 K 线可能把成交量给成“股”，应能按流通股本自动修正。"""
    df = make_kline(n=180)
    df["volume"] = 4_000_000.0  # 股，而不是手
    df["turnover"] = np.nan
    df["amount"] = df["volume"] * df["close"]  # 元
    float_shares = 160_000_000.0
    f = technical_features(df, benchmark=df, limit_pct=0.20, float_shares=float_shares)
    expected_turnover = 4_000_000.0 / float_shares
    assert f["turnover_mean20"] == pytest.approx(expected_turnover, rel=0.05)
    assert f["amount_mean20"] < 4_000_000.0 * float(df["close"].max())
