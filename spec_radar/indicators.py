# -*- coding: utf-8 -*-
"""量价特征工程：只使用信号日及之前的数据，避免未来函数。"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

EPS = 1e-12


def _ret(close: np.ndarray, n: int) -> float:
    if len(close) <= n or close[-1 - n] <= 0:
        return np.nan
    return float(close[-1] / close[-1 - n] - 1.0)


def _ma(close: np.ndarray, n: int) -> float:
    if len(close) < n:
        return np.nan
    return float(np.mean(close[-n:]))


def _slope(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * (values - values.mean())) / denom)


def _rsi(close: np.ndarray, n: int = 14) -> float:
    if len(close) < n + 1:
        return np.nan
    diff = np.diff(close[-(n + 1):])
    gains = np.clip(diff, 0, None)
    losses = -np.clip(diff, None, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= EPS:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    peak = np.maximum.accumulate(values)
    dd = values / np.maximum(peak, EPS) - 1.0
    return float(np.min(dd))


def _last_finite(value: float, default: float = np.nan) -> float:
    try:
        return float(value) if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def technical_features(
    frame: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    limit_pct: float = 0.10,
    float_shares: Optional[float] = None,
) -> Dict[str, float]:
    """从单只股票的日 K 计算信号日特征。

    frame 必须包含 date/open/close/high/low/volume/amount/turnover/pct。
    """
    if frame is None or frame.empty:
        raise ValueError("空 K 线")
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close", "high", "low"])
    df = df[df["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"K 线不足：{len(df)} 根")

    for col in ("open", "close", "high", "low", "volume", "amount", "turnover", "pct"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    # 腾讯备用源没有成交额/换手率：用均价成交额、流通股本还原，足以做相对排序。
    if df["amount"].isna().all():
        df["amount"] = df["volume"] * 100.0 * (df["open"].fillna(df["close"]) + df["close"]) / 2.0
    else:
        df["amount"] = df["amount"].fillna(df["volume"] * 100.0 * df["close"])
    # 备用源（尤其腾讯科创板/创业板）有时把成交量给成“股”而不是“手”。
    # 用流通股本反推换手：若近 60 日中位换手 > 200%，几乎可以确定单位是股，整体除以 100。
    if float_shares and float_shares > 0 and len(df):
        raw_turnover = df["volume"].to_numpy(dtype=float) * 100.0 / float(float_shares)
        recent = raw_turnover[-60:]
        if np.isfinite(recent).any() and float(np.nanmedian(recent)) > 2.0:
            df["volume"] = df["volume"] / 100.0
            df["amount"] = df["amount"] / 100.0
    if df["turnover"].isna().all() and float_shares and float_shares > 0:
        df["turnover"] = df["volume"] * 100.0 / float(float_shares) * 100.0
    df["turnover"] = df["turnover"].fillna(0.0)
    df["pct"] = df["pct"].fillna(df["close"].pct_change() * 100.0)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float)
    turnover = df["turnover"].to_numpy(dtype=float) / 100.0
    pct = df["pct"].to_numpy(dtype=float) / 100.0
    n = len(close)

    out: Dict[str, float] = {"bars": float(n)}
    for horizon in (1, 5, 10, 20, 60, 120, 250):
        out[f"ret{horizon}"] = _ret(close, horizon)
    if not np.isfinite(out["ret250"]) and n >= 120:
        out["ret250"] = _ret(close, n - 1)

    for window in (5, 10, 20, 60, 120, 250):
        out[f"ma{window}"] = _ma(close, window)
        if np.isfinite(out[f"ma{window}"]):
            out[f"dist_ma{window}"] = float(close[-1] / out[f"ma{window}"] - 1.0)
        else:
            out[f"dist_ma{window}"] = np.nan
    out["ma20_slope"] = (
        float(out["ma20"] / _ma(close[:-10], 20) - 1.0)
        if np.isfinite(out["ma20"]) and n > 30
        else np.nan
    )
    out["ma60_slope"] = (
        float(out["ma60"] / _ma(close[:-20], 60) - 1.0)
        if np.isfinite(out["ma60"]) and n > 80
        else np.nan
    )

    out["above_ma20"] = float(close[-1] > out["ma20"]) if np.isfinite(out["ma20"]) else np.nan
    out["above_ma60"] = float(close[-1] > out["ma60"]) if np.isfinite(out["ma60"]) else np.nan
    out["above_ma120"] = float(close[-1] > out["ma120"]) if np.isfinite(out["ma120"]) else np.nan
    out["above_ma250"] = float(close[-1] > out["ma250"]) if np.isfinite(out["ma250"]) else np.nan
    out["trend_align"] = float(
        sum([
            np.isfinite(out["ma5"]) and close[-1] > out["ma5"],
            np.isfinite(out["ma10"]) and np.isfinite(out["ma5"]) and out["ma5"] > out["ma10"],
            np.isfinite(out["ma20"]) and np.isfinite(out["ma10"]) and out["ma10"] > out["ma20"],
            np.isfinite(out["ma60"]) and np.isfinite(out["ma20"]) and out["ma20"] > out["ma60"],
            np.isfinite(out["ma120"]) and np.isfinite(out["ma60"]) and out["ma60"] > out["ma120"],
        ])
    )

    rets = np.diff(close) / np.maximum(close[:-1], EPS)
    if len(rets) >= 20:
        out["vol20"] = float(np.std(rets[-20:], ddof=1) * math.sqrt(252))
    else:
        out["vol20"] = np.nan
    if len(rets) >= 60:
        out["vol60"] = float(np.std(rets[-60:], ddof=1) * math.sqrt(252))
    else:
        out["vol60"] = np.nan
    if np.isfinite(out["vol20"]) and np.isfinite(out["vol60"]) and out["vol60"] > EPS:
        out["vol_ratio_20_60"] = float(out["vol20"] / out["vol60"])
    else:
        out["vol_ratio_20_60"] = np.nan
    if len(rets) >= 120:
        vol120 = float(np.std(rets[-120:], ddof=1) * math.sqrt(252))
        out["vol_ratio_20_120"] = float(out["vol20"] / vol120) if vol120 > EPS else np.nan
    else:
        out["vol_ratio_20_120"] = np.nan

    # ATR 与布林带宽度
    if n >= 15:
        prev_close = close[:-1]
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ])
        out["atr14"] = float(np.mean(tr[-14:]) / close[-1])
    else:
        out["atr14"] = np.nan
    if n >= 20 and np.isfinite(out["ma20"]):
        out["bb_width"] = float(4.0 * np.std(close[-20:], ddof=1) / max(out["ma20"], EPS))
    else:
        out["bb_width"] = np.nan
    out["rsi14"] = _rsi(close, 14)

    if n >= 20:
        out["range20"] = float((high[-20:].max() - low[-20:].min()) / max(close[-1], EPS))
    else:
        out["range20"] = np.nan
    if n >= 60:
        out["range60"] = float((high[-60:].max() - low[-60:].min()) / max(close[-1], EPS))
        out["dist_high60"] = float(close[-1] / max(high[-60:].max(), EPS) - 1.0)
        out["dist_low60"] = float(close[-1] / max(low[-60:].min(), EPS) - 1.0)
    else:
        out["range60"] = np.nan
        out["dist_high60"] = np.nan
        out["dist_low60"] = np.nan
    if n >= 120:
        out["dist_high120"] = float(close[-1] / max(high[-120:].max(), EPS) - 1.0)
    else:
        out["dist_high120"] = np.nan
    lookback = min(n, 250)
    hi = float(high[-lookback:].max())
    lo = float(low[-lookback:].min())
    out["high250"] = hi
    out["low250"] = lo
    out["dd52"] = float(close[-1] / max(hi, EPS) - 1.0)
    out["pos52"] = float((close[-1] - lo) / max(hi - lo, EPS)) if hi > lo else 0.5
    out["days_since_high"] = float(lookback - 1 - int(np.argmax(high[-lookback:])))
    out["maxdd60"] = _max_drawdown(close[-60:]) if n >= 20 else np.nan
    out["maxdd120"] = _max_drawdown(close[-120:]) if n >= 40 else np.nan
    out["vcp_ratio"] = (
        float(out["range20"] / out["range60"])
        if np.isfinite(out["range20"]) and np.isfinite(out["range60"]) and out["range60"] > EPS
        else np.nan
    )

    prev_high20 = float(high[-21:-1].max()) if n >= 21 else np.nan
    prev_high60 = float(high[-61:-1].max()) if n >= 61 else np.nan
    out["breakout20"] = float(close[-1] > prev_high20) if np.isfinite(prev_high20) else np.nan
    out["breakout60"] = float(close[-1] > prev_high60) if np.isfinite(prev_high60) else np.nan
    out["near_high60"] = float(close[-1] / max(prev_high60, EPS)) if np.isfinite(prev_high60) else np.nan
    out["near_high250"] = float(close[-1] / max(hi, EPS)) if hi > 0 else np.nan
    day_range = high[-1] - low[-1]
    out["close_pos_day"] = float((close[-1] - low[-1]) / day_range) if day_range > EPS else 0.5
    out["gap_open"] = float(open_[-1] / close[-2] - 1.0) if n >= 2 else np.nan

    # 成交与换手
    for window in (5, 20, 60, 120):
        out[f"amount_mean{window}"] = float(np.nanmean(amount[-window:])) if n >= window else np.nan
        out[f"turnover_mean{window}"] = float(np.nanmean(turnover[-window:])) if n >= window else np.nan
        out[f"volume_mean{window}"] = float(np.nanmean(volume[-window:])) if n >= window else np.nan
    out["amount_ratio_5_20"] = (
        out["amount_mean5"] / out["amount_mean20"]
        if np.isfinite(out["amount_mean20"]) and out["amount_mean20"] > EPS else np.nan
    )
    out["amount_ratio_20_60"] = (
        out["amount_mean20"] / out["amount_mean60"]
        if np.isfinite(out["amount_mean60"]) and out["amount_mean60"] > EPS else np.nan
    )
    out["amount_ratio_5_60"] = (
        out["amount_mean5"] / out["amount_mean60"]
        if np.isfinite(out["amount_mean60"]) and out["amount_mean60"] > EPS else np.nan
    )
    out["amount_ratio_20_120"] = (
        out["amount_mean20"] / out["amount_mean120"]
        if np.isfinite(out["amount_mean120"]) and out["amount_mean120"] > EPS else np.nan
    )
    out["turnover_ratio_5_20"] = (
        out["turnover_mean5"] / out["turnover_mean20"]
        if np.isfinite(out["turnover_mean20"]) and out["turnover_mean20"] > EPS else np.nan
    )
    out["volume_expansion"] = out["amount_ratio_5_20"]

    # OBV / CMF / 上涨日量能
    direction = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(direction * volume)
    if n >= 21:
        obv_slope = _slope(obv[-20:])
        avg_vol = float(np.nanmean(volume[-20:]))
        out["obv_slope20"] = float(obv_slope / max(avg_vol, EPS)) if np.isfinite(obv_slope) else np.nan
        price_slope = _slope(close[-20:])
        out["price_slope20"] = float(price_slope / max(float(np.nanmean(close[-20:])), EPS)) if np.isfinite(price_slope) else np.nan
        out["obv_divergence"] = (
            out["obv_slope20"] - out["price_slope20"]
            if np.isfinite(out["obv_slope20"]) and np.isfinite(out["price_slope20"]) else np.nan
        )
    else:
        out["obv_slope20"] = np.nan
        out["price_slope20"] = np.nan
        out["obv_divergence"] = np.nan

    if n >= 20:
        mfm = (2.0 * close[-20:] - high[-20:] - low[-20:]) / np.maximum(high[-20:] - low[-20:], EPS)
        mfm = np.clip(mfm, -1.0, 1.0)
        denom = float(np.nansum(volume[-20:]))
        out["cmf20"] = float(np.nansum(mfm * volume[-20:]) / denom) if denom > EPS else np.nan
        up_mask = pct[-20:] > 0
        down_mask = pct[-20:] < 0
        up_vol = float(np.nansum(volume[-20:][up_mask]))
        down_vol = float(np.nansum(volume[-20:][down_mask]))
        out["up_down_vol_ratio"] = up_vol / max(down_vol, EPS) if up_vol > 0 else np.nan
        out["up_days20"] = float(up_mask.sum())
        vol_chg20 = np.diff(volume[-21:]) / np.maximum(volume[-21:-1], EPS) if n >= 21 else np.array([])
        out["vol_price_corr20"] = (
            float(np.corrcoef(pct[-20:], vol_chg20)[0, 1])
            if len(vol_chg20) == 20 and np.std(vol_chg20) > 0 and np.std(pct[-20:]) > 0 else np.nan
        )
    else:
        out["cmf20"] = np.nan
        out["up_down_vol_ratio"] = np.nan
        out["up_days20"] = np.nan
        out["vol_price_corr20"] = np.nan

    # 涨跌停/极端日
    lim = float(limit_pct)
    limit_flags = pct >= lim - 0.0025
    out["limit_up20"] = float(np.nansum(limit_flags[-20:])) if n >= 20 else np.nan
    out["limit_up60"] = float(np.nansum(limit_flags[-60:])) if n >= 60 else np.nan
    out["limit_down20"] = float(np.nansum(pct[-20:] <= -lim + 0.0025)) if n >= 20 else np.nan
    out["big_down5"] = float(np.nansum(pct[-5:] <= -0.05)) if n >= 5 else np.nan
    out["big_up5"] = float(np.nansum(pct[-5:] >= 0.05)) if n >= 5 else np.nan
    out["max_day_gain20"] = float(np.nanmax(pct[-20:])) if n >= 20 else np.nan
    out["limit_up_today"] = float(limit_flags[-1]) if n else np.nan
    out["prev_limit_up"] = float(limit_flags[-2]) if n >= 2 else np.nan
    board_count = 0
    for flag in limit_flags[::-1]:
        if bool(flag):
            board_count += 1
        else:
            break
    out["consecutive_limit_up"] = float(min(board_count, 20))
    out["first_board"] = float(bool(limit_flags[-1]) and not bool(limit_flags[-2])) if n >= 2 else np.nan
    out["limit_up_days10"] = float(np.nansum(limit_flags[-10:])) if n >= 10 else np.nan
    prev_close = close[-2] if n >= 2 else close[-1]
    out["amplitude_today"] = float((high[-1] - low[-1]) / max(prev_close, EPS))
    out["turnover_today"] = float(turnover[-1]) if n else np.nan
    out["volume_ratio_5"] = (
        float(volume[-1] / max(float(np.nanmean(volume[-6:-1])), EPS)) if n >= 6 else np.nan
    )

    # MACD 柱（归一化）
    if n >= 35:
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy()
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().to_numpy()
        out["macd_hist"] = float((dif[-1] - dea[-1]) / max(close[-1], EPS))
        out["macd_hist_prev"] = float((dif[-6] - dea[-6]) / max(close[-6], EPS)) if n >= 40 else np.nan
    else:
        out["macd_hist"] = np.nan
        out["macd_hist_prev"] = np.nan

    # 与基准的 beta / 相关性
    if benchmark is not None and not benchmark.empty and n >= 60:
        bench = benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
        bench = bench.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
        bench_s = bench.set_index("date")["close"].reindex(df["date"]).ffill()
        bret = bench_s.pct_change().to_numpy(dtype=float)
        pair = pd.DataFrame({"r": rets, "b": bret[1:] if len(bret) == len(rets) + 1 else bret[:len(rets)]}).dropna().tail(60)
        if len(pair) >= 20 and float(pair["b"].var()) > EPS:
            cov = float(np.cov(pair["r"], pair["b"], ddof=1)[0, 1])
            out["beta60"] = cov / float(pair["b"].var())
            out["corr60"] = float(pair["r"].corr(pair["b"]))
        else:
            out["beta60"] = np.nan
            out["corr60"] = np.nan
    else:
        out["beta60"] = np.nan
        out["corr60"] = np.nan

    out["close_signal"] = float(close[-1])
    out["date_signal"] = pd.Timestamp(df["date"].iloc[-1])
    out["float_shares"] = float(float_shares) if float_shares and float_shares > 0 else np.nan
    out["avg_amount20_yi"] = out["amount_mean20"] / 1e8 if np.isfinite(out["amount_mean20"]) else np.nan
    out["limit_pct"] = lim
    out["last_day_pct"] = float(pct[-1]) if n else np.nan
    out["last_amount_yi"] = float(amount[-1] / 1e8) if n else np.nan
    # 用有限的默认值替换 inf，排名层再做中性填充。
    return {k: _last_finite(v, np.nan) if not isinstance(v, pd.Timestamp) else v for k, v in out.items()}


def compute_feature_frame(
    panel: pd.DataFrame,
    benchmark: Optional[pd.DataFrame],
    snapshot: pd.DataFrame,
    min_bars: int = 90,
) -> pd.DataFrame:
    """把长表 K 线转换为每只股票一行的特征表。"""
    if panel.empty:
        return pd.DataFrame()
    meta = snapshot.set_index("code")[["name", "industry", "board", "limit_pct", "float_shares"]].to_dict("index")
    rows = []
    for code, group in panel.groupby("code", sort=False):
        info = meta.get(str(code), {})
        try:
            feat = technical_features(
                group,
                benchmark=benchmark,
                limit_pct=float(info.get("limit_pct", 0.10) or 0.10),
                float_shares=info.get("float_shares"),
            )
        except Exception:
            continue
        if feat.get("bars", 0) < min_bars:
            continue
        feat["code"] = str(code).zfill(6)
        rows.append(feat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("code", drop=False)
