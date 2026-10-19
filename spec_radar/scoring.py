# -*- coding: utf-8 -*-
"""双轨打分模型：潜在投机 vs 活跃投机。

所有指标都在“信号日”截面上做横截面百分位，再做可解释加权；最后减去绝对风险罚分。
不使用未来数据；潜在模型会硬性排除已经充分交易/连续涨停的标的，活跃模型保留强趋势但惩罚过热。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ScanSettings
from .macro import external_risk_off_score
from .themes import ThemeMapper

COMPONENT_LABELS = {
    "policy": "赛道政策催化",
    "compression": "波动压缩蓄势",
    "accumulation": "资金吸筹",
    "base": "位置与赔率",
    "fundamental": "基本面拐点",
    "neglect": "关注度低位",
    "elasticity": "投机弹性",
    "regime": "市场环境匹配",
    "momentum": "动量强度",
    "theme": "主题热度/龙头",
    "volume": "量能与流动性",
    "breakout": "突破与趋势",
    "flow": "资金流",
}
COMPONENT_ORDER = {
    "potential": ["policy", "compression", "accumulation", "base", "fundamental", "neglect", "elasticity", "regime"],
    "active": ["momentum", "theme", "volume", "breakout", "flow", "elasticity", "fundamental", "regime"],
}
COMPONENT_NOTES = {
    "policy": "行业/概念与当前政策主线、全球宏观情景的契合度，兼顾“强主题中的滞涨者”。",
    "compression": "ATR、60日波幅、布林带宽、量能萎缩与VCP收缩，寻找蓄势而非破位。",
    "accumulation": "OBV斜率、CMF、价量背离与上涨日量能，识别未被价格充分反映的吸筹。",
    "base": "距52周高点的甜点回撤、MA120/250支撑、横盘时长与稳定性，避免接飞刀。",
    "fundamental": "营收/利润增速、ROE、毛利率与估值/盈利质量；权重不高，用于排除纯壳。",
    "neglect": "低换手、低短期涨幅、少涨停、低概念热度，捕捉尚未被充分关注的标的。",
    "elasticity": "流通市值、Beta与20%涨跌停板属性，衡量题材启动后的价格弹性。",
    "regime": "风险偏好、市场宽度、波动和外部避险条件下，该风格是否顺风。",
    "momentum": "5/20/60日收益、加速度、MA20斜率与MACD，确认资金已经选择方向。",
    "theme": "政策主题契合、概念板块热度、行业相对强度与龙头地位。",
    "volume": "成交额、换手、量能扩张和上涨日量比，确认投机资金真实参与。",
    "breakout": "距60日/52周高点、突破形态、均线多头排列与相对强度。",
    "flow": "OBV、CMF、量价背离与上涨日量能，判断资金流方向。",
}


def _rank(s: pd.Series, high_is_good: bool = True, fill: float = 35.0) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 3:
        return pd.Series(50.0, index=s.index)
    pct = x.rank(method="average", pct=True, na_option="keep")
    if not high_is_good:
        pct = 1.0 - pct
    return (pct * 100.0).fillna(fill)


def _clip(x: Any, lo: float = 0.0, hi: float = 100.0) -> Any:
    return np.clip(x, lo, hi)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _size_sweet(cap: pd.Series, low: float, high: float) -> pd.Series:
    """流通市值甜点区：太小吃流动性/控盘风险，太大拉不动。"""
    x = pd.to_numeric(cap, errors="coerce").clip(lower=1)
    center = float(np.sqrt(low * high))
    half_width = max(0.15, float(np.log(high / low)) / 2.0)
    score = (1.0 - (np.log(x / center).abs() / half_width)).clip(0, 1) * 100.0
    return score.fillna(30.0)


def _pe_score(pe: Any) -> float:
    v = _finite(pe)
    if not np.isfinite(v):
        return 40.0
    if v <= 0:
        return 20.0
    if v <= 25:
        return 100.0
    if v <= 45:
        return 82.0
    if v <= 80:
        return 62.0
    if v <= 150:
        return 35.0
    return 15.0


def _age_days(snapshot: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(snapshot["list_date"], errors="coerce")
    return (signal_date - dates).dt.days


def base_eligibility(snapshot: pd.DataFrame, settings: ScanSettings, signal_date: pd.Timestamp) -> pd.DataFrame:
    """与模型无关的基础可交易域。"""
    df = snapshot.copy()
    df["age_days"] = _age_days(df, signal_date)
    mask = (
        (~df["is_st"])
        & (~df["is_suspended"])
        & (df["board"] != "北交所")
        & (df["board"].isin(list(settings.boards)))
        & (df["price"] >= 2.0)
        & (df["age_days"] >= 150)
        & (df["float_mcap_yi"] >= 8.0)
        & (df["float_mcap_yi"] <= 5000.0)
        & (df["amount"] >= 2e7)
    )
    return df.loc[mask].copy()


def prefilter_codes(
    snapshot: pd.DataFrame,
    settings: ScanSettings,
    model_cfg: Dict[str, Any],
    mapper: ThemeMapper,
    signal_date: pd.Timestamp | None = None,
    adjust_live_snapshot: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """用快照做粗筛，只对最有希望的股票抓取日 K，控制云端耗时。

    若信号日是上一交易日、快照却是盘中数据，则先剔除当日涨跌，近似还原信号日收盘的
    60 日/年初至今涨幅，避免粗筛阶段偷看当天行情。
    """
    if signal_date is not None:
        signal_ts = pd.Timestamp(signal_date)
    else:
        signal_ts = pd.Timestamp(settings.as_of) if settings.as_of else pd.Timestamp.today().normalize()
    base = base_eligibility(snapshot, settings, signal_ts)
    if base.empty:
        return [], {"base": 0, "potential": 0, "active": 0}
    # 主题暴露：用行业 + 概念映射；为控制耗时，只对基础域计算。
    theme_fit, theme_tilt = [], []
    for _, row in base.iterrows():
        match = mapper.match(str(row.get("industry", "")), list(row.get("concepts", [])))
        theme_fit.append(match.fit)
        theme_tilt.append(max(match.tilt, 0.0))
    base = base.copy()
    base["pref_theme_fit"] = theme_fit
    base["pref_theme_tilt"] = theme_tilt

    ret60_raw = pd.to_numeric(base["ret60_snap"], errors="coerce")
    ytd_raw = pd.to_numeric(base["ret_ytd_snap"], errors="coerce")
    day_pct = pd.to_numeric(base["day_pct_snap"], errors="coerce").fillna(0.0)
    if adjust_live_snapshot:
        denom = (1.0 + day_pct).clip(lower=0.2)
        ret60 = (1.0 + ret60_raw) / denom - 1.0
        ytd = (1.0 + ytd_raw) / denom - 1.0
    else:
        ret60 = ret60_raw
        ytd = ytd_raw
    amount = pd.to_numeric(base["amount"], errors="coerce")
    size = pd.to_numeric(base["float_mcap_yi"], errors="coerce")
    amount_sweet = np.exp(-((np.log(amount.clip(lower=1e6)) - np.log(2.5e8)) / 1.15) ** 2)
    ytd_pot = (1.0 - np.clip((ytd.fillna(0) - 0.60) / 0.80, 0, 1)).clip(0, 1)

    theme_pref = base["pref_theme_fit"] * base["pref_theme_tilt"] * 100
    pot = (
        0.30 * _rank(ret60, high_is_good=False)
        + 0.20 * (ytd_pot * 100)
        + 0.15 * _rank(size, high_is_good=False)
        + 0.20 * theme_pref
        + 0.15 * (amount_sweet * 100)
    )
    active = (
        0.35 * _rank(ret60, high_is_good=True)
        + 0.20 * _rank(ytd, high_is_good=True)
        + 0.20 * theme_pref
        + 0.15 * _rank(amount, high_is_good=True)
        + 0.10 * _rank(size, high_is_good=False)
    )
    base["pref_potential"] = pot
    base["pref_active"] = active
    n_pot = min(int(settings.scan_potential), len(base))
    n_act = min(int(settings.scan_active), len(base))
    top_pot = set(base.nlargest(n_pot, "pref_potential")["code"].astype(str))
    top_act = set(base.nlargest(n_act, "pref_active")["code"].astype(str))
    codes = sorted(top_pot | top_act)
    return codes, {"base": int(len(base)), "potential": len(top_pot), "active": len(top_act)}


def _model_eligibility(
    features: pd.DataFrame,
    snapshot: pd.DataFrame,
    model_key: str,
    settings: ScanSettings,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    cfg = settings  # 用户覆盖
    model = _MODEL_CACHE["cfg"][model_key]
    elig = model.get("eligibility", {})
    df = features.copy()
    df["code"] = df.index.astype(str).str.zfill(6)
    snap = snapshot.set_index("code")
    meta_cols = ["name", "board", "price", "float_mcap_yi", "pe_ttm", "pb", "net_profit",
                 "roe", "profit_yoy", "rev_yoy", "gross_margin", "main_inflow_yi",
                 "concepts", "region", "list_date", "is_st"]
    meta = snap.reindex(df["code"])[meta_cols].reset_index(drop=True)
    for col in meta.columns:
        df[col] = meta[col].values
    df["age_days"] = (signal_date - pd.to_datetime(df["list_date"], errors="coerce")).dt.days
    min_cap = max(float(elig.get("min_float_mcap_yi", 0)), float(cfg.min_float_mcap_yi or 0))
    max_cap = float(elig.get("max_float_mcap_yi", 1e9))
    if cfg.max_float_mcap_yi and cfg.max_float_mcap_yi > 0:
        max_cap = min(max_cap, float(cfg.max_float_mcap_yi))
    min_amt = max(float(elig.get("min_amount20_yi", 0)), float(cfg.min_amount20_yi or 0))
    mask = (
        (~df["is_st"])
        & (df["board"].isin(list(settings.boards)))
        & (df["close_signal"].fillna(df["price"]) >= float(elig.get("min_price", 2.0)))
        & (df["age_days"] >= int(elig.get("min_list_days", 150)))
        & (df["bars"] >= int(elig.get("min_bars", 90)))
        & (df["float_mcap_yi"] >= min_cap)
        & (df["float_mcap_yi"] <= max_cap)
        & (df["avg_amount20_yi"] >= min_amt)
    )
    if settings.exclude_loss:
        mask &= df["net_profit"].fillna(-1) > 0
    if model_key == "potential":
        mask &= df["ret60"].fillna(0) <= float(elig.get("max_ret60", 1.0))
        mask &= df["ret20"].fillna(0) <= float(elig.get("max_ret20", 1.0))
        mask &= df["turnover_mean20"].fillna(0) <= float(elig.get("max_turnover20", 1.0))
        mask &= df["limit_up20"].fillna(0) <= int(elig.get("max_limit_up20", 99))
    else:
        mask &= df["ret20"].fillna(-1) >= float(elig.get("min_ret20", -1.0))
        mask &= df["turnover_mean20"].fillna(0) >= float(elig.get("min_turnover20", 0.0))
        if elig.get("require_above_ma20", False):
            mask &= df["close_signal"].fillna(0) >= df["ma20"].fillna(np.inf)
    return df.loc[mask].copy()


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _set_model_config(model_cfg: Dict[str, Any]) -> None:
    _MODEL_CACHE["cfg"] = model_cfg


def _components_potential(df: pd.DataFrame, macro: Dict[str, Any]) -> pd.DataFrame:
    c = pd.DataFrame(index=df.index)
    theme_exposure = (df["theme_fit"].fillna(0) * df["theme_tilt"].clip(lower=0).fillna(0) * 100.0).clip(0, 100)
    ind_rank = _rank(df["ind_ret20"], True)
    lag_bonus = pd.Series(45.0, index=df.index)
    has_theme = df["theme_fit"].fillna(0) > 0.2
    lag_bonus.loc[has_theme & (df["rs20"] < -0.01) & (df["ret20"] > -0.15)] = 80.0
    lag_bonus.loc[has_theme & (df["rs20"] > 0.15)] = 25.0
    theme_ret = df["theme_proxy_ret20"].fillna(0)
    timing = pd.Series(50.0, index=df.index)
    timing.loc[has_theme] = (100.0 - np.clip((theme_ret.loc[has_theme] - 0.03) / 0.20, 0, 1) * 55.0)
    c["policy"] = _clip(0.52 * theme_exposure + 0.18 * ind_rank + 0.15 * lag_bonus + 0.15 * timing)

    comp = (
        0.24 * _rank(df["atr14"], False)
        + 0.18 * _rank(df["range60"], False)
        + 0.18 * _rank(df["bb_width"], False)
        + 0.14 * _rank(df["vol_ratio_20_120"], False)
        + 0.12 * _rank(df["vcp_ratio"], False)
        + 0.14 * _rank(df["amount_ratio_20_120"], False)
    )
    c["compression"] = _clip(comp)

    acc = (
        0.30 * _rank(df["obv_slope20"], True)
        + 0.25 * _rank(df["cmf20"], True)
        + 0.20 * _rank(df["obv_divergence"], True)
        + 0.15 * _rank(df["up_down_vol_ratio"], True)
        + 0.10 * _rank(df["amount_ratio_5_20"], True)
    )
    c["accumulation"] = _clip(acc)

    dd = df["dd52"].fillna(-0.30)
    dd_score = (1.0 - (dd + 0.22).abs() / 0.38).clip(0, 1) * 100
    above120 = df["above_ma120"].fillna(0)
    above250 = df["above_ma250"].fillna(0)
    support = 20.0 + 40.0 * above120 + 40.0 * above250
    support = pd.Series(support, index=df.index).clip(0, 100)
    base_len = np.clip(df["days_since_high"].fillna(0) / 120.0, 0, 1) * 100
    slope = np.clip((df["ma20_slope"].fillna(0) + 0.03) / 0.06, 0, 1) * 100
    stability = _rank(df["maxdd60"], False)
    lag_neutral = _rank(df["rs20"].abs(), False)
    c["base"] = _clip(
        0.30 * dd_score + 0.20 * support + 0.15 * base_len
        + 0.15 * slope + 0.10 * stability + 0.10 * lag_neutral
    )

    growth = (
        0.40 * _rank(df["profit_yoy"], True)
        + 0.25 * _rank(df["rev_yoy"], True)
        + 0.20 * _rank(df["roe"], True)
        + 0.15 * _rank(df["gross_margin"], True)
    )
    pe_score = df["pe_ttm"].map(_pe_score)
    profit_pos = np.where(df["net_profit"].fillna(-1) > 0, 85.0, 30.0)
    c["fundamental"] = _clip(0.60 * growth + 0.25 * pe_score + 0.15 * profit_pos)

    neglect = (
        0.30 * _rank(df["turnover_mean20"], False)
        + 0.25 * _rank(df["ret20"].abs(), False)
        + 0.20 * _rank(df["limit_up20"], False)
        + 0.15 * _rank(df["concept_day"], False)
        + 0.10 * _rank(df["concept_60d"], False)
    )
    c["neglect"] = _clip(neglect)

    elasticity = (
        0.40 * _size_sweet(df["float_mcap_yi"], 50.0, 120.0)
        + 0.20 * _rank(df["float_mcap_yi"], False)
        + 0.22 * _rank(df["beta60"].abs(), True)
        + 0.18 * np.where(df["limit_pct"] >= 0.195, 95.0, 65.0)
    )
    c["elasticity"] = _clip(elasticity)

    risk = float(macro.get("risk_appetite", 50))
    ext_off = max(0.0, external_risk_off_score(macro))
    small = _rank(df["float_mcap_yi"], False)
    low_vol = _rank(df["vol20"], False)
    quality = _rank(df["profit_yoy"], True)
    defensive = (df["theme_tilt"].fillna(0) <= 0.5) & (df["theme_fit"].fillna(0) > 0)
    defensive_score = np.where(defensive, 70.0, 45.0)
    if risk >= 60:
        regime = 50 + 0.22 * (small - 50) + 0.15 * (c["compression"] - 50) + 0.10 * (ind_rank - 50)
    elif risk <= 42:
        regime = 50 + 0.28 * (low_vol - 50) + 0.24 * (quality - 50) + 0.16 * (pd.Series(defensive_score, index=df.index) - 50)
    else:
        regime = 50 + 0.12 * (small - 50) + 0.12 * (quality - 50) + 0.10 * (low_vol - 50)
    regime = pd.Series(regime, index=df.index) - 12.0 * ext_off
    c["regime"] = _clip(regime)
    return c


def _components_active(df: pd.DataFrame, macro: Dict[str, Any]) -> pd.DataFrame:
    c = pd.DataFrame(index=df.index)
    accel = df["ret20"].fillna(0) - df["ret60"].fillna(0) / 3.0
    momentum = (
        0.18 * _rank(df["ret5"], True)
        + 0.30 * _rank(df["ret20"], True)
        + 0.17 * _rank(df["ret60"], True)
        + 0.15 * _rank(accel, True)
        + 0.12 * _rank(df["ma20_slope"], True)
        + 0.08 * _rank(df["macd_hist"], True)
    )
    c["momentum"] = _clip(momentum)

    theme_exposure = (df["theme_fit"].fillna(0) * df["theme_tilt"].clip(lower=0.1).fillna(0) * 100.0).clip(0, 100)
    concept_heat = (
        0.45 * _rank(df["concept_day"], True)
        + 0.30 * _rank(df["concept_60d"], True)
        + 0.25 * _rank(df.get("concept_20d", pd.Series(np.nan, index=df.index)), True)
    )
    industry_heat = (
        0.45 * _rank(df["ind_ret20"], True)
        + 0.25 * _rank(df.get("ind_day", pd.Series(np.nan, index=df.index)), True)
        + 0.30 * _rank(df["ind_breadth"], True)
    )
    leadership = (
        0.50 * _rank(df["rs20"], True)
        + 0.30 * _rank(df["rs60"], True)
        + 0.20 * _rank(df["limit_up20"], True)
    )
    sector_limitup = df.get("sector_limitup_count", pd.Series(np.nan, index=df.index))
    sector_heat = _rank(sector_limitup, True)
    board_count = df.get("consecutive_limit_up", pd.Series(0.0, index=df.index)).fillna(0)
    board_score = pd.Series(50.0, index=df.index)
    board_score.loc[board_count <= 0] = 35.0
    board_score.loc[(board_count >= 1) & (board_count <= 3)] = 90.0
    board_score.loc[(board_count >= 4) & (board_count <= 5)] = 65.0
    board_score.loc[board_count >= 6] = 30.0
    first_board = df.get("first_board", pd.Series(0.0, index=df.index)).fillna(0)
    recognition = 0.55 * board_score + 0.25 * _rank(df["limit_up20"], True) + 0.20 * (first_board * 100.0)
    c["theme"] = _clip(
        0.26 * theme_exposure + 0.19 * concept_heat + 0.14 * industry_heat
        + 0.13 * leadership + 0.18 * recognition + 0.10 * sector_heat
    )

    seal_ratio = df.get("seal_ratio", pd.Series(np.nan, index=df.index))
    seal_quality = (seal_ratio.fillna(0) / 0.03).clip(0, 1) * 100
    open_count = df.get("open_count", pd.Series(np.nan, index=df.index))
    volume = (
        0.28 * _rank(np.log1p(df["amount_mean5"].clip(lower=0)), True)
        + 0.22 * _rank(df["turnover_mean5"], True)
        + 0.22 * _rank(df["volume_expansion"], True)
        + 0.18 * _rank(df["up_down_vol_ratio"], True)
        + 0.10 * seal_quality
    )
    volume = volume - 8.0 * (open_count.fillna(0) >= 3).astype(float)
    c["volume"] = _clip(volume)

    near60 = np.clip((1.0 + df["dist_high60"].fillna(-0.25) - 0.75) / 0.25, 0, 1) * 100
    pos52 = df["pos52"].fillna(0.5) * 100
    trend = df["trend_align"].fillna(0) / 5.0 * 100
    breakout = pd.Series(25.0, index=df.index)
    breakout.loc[(df["breakout20"].fillna(0) > 0) | (df["near_high60"].fillna(0) >= 0.98)] = 80.0
    breakout.loc[df["breakout60"].fillna(0) > 0] = 100.0
    c["breakout"] = _clip(
        0.25 * near60 + 0.20 * pos52 + 0.20 * trend
        + 0.20 * breakout + 0.15 * _rank(df["rs20"], True)
    )

    lhb_net = df.get("lhb_net_yi", pd.Series(np.nan, index=df.index))
    lhb_hot = df.get("lhb_hot_seat_count", pd.Series(np.nan, index=df.index))
    lhb_score = 50.0 + 25.0 * np.tanh(lhb_net.fillna(0) / 0.8) + 10.0 * np.clip(lhb_hot.fillna(0), 0, 3)
    flow = (
        0.32 * _rank(df["obv_slope20"], True)
        + 0.27 * _rank(df["cmf20"], True)
        + 0.18 * _rank(df["up_down_vol_ratio"], True)
        + 0.13 * _rank(df["obv_divergence"], True)
        + 0.10 * pd.Series(lhb_score, index=df.index).clip(0, 100)
    )
    c["flow"] = _clip(flow)

    elasticity = (
        0.45 * _size_sweet(df["float_mcap_yi"], 30.0, 150.0)
        + 0.18 * _rank(df["float_mcap_yi"], False)
        + 0.22 * _rank(df["beta60"].abs(), True)
        + 0.15 * np.where(df["limit_pct"] >= 0.195, 95.0, 65.0)
    )
    c["elasticity"] = _clip(elasticity)

    fundamental = (
        0.45 * _rank(df["profit_yoy"], True)
        + 0.25 * _rank(df["rev_yoy"], True)
        + 0.15 * _rank(df["roe"], True)
        + 0.15 * df["pe_ttm"].map(_pe_score)
    )
    c["fundamental"] = _clip(fundamental)

    risk = float(macro.get("risk_appetite", 50))
    ext_off = max(0.0, external_risk_off_score(macro))
    beta_rank = _rank(df["beta60"].abs(), True)
    quality = _rank(df["profit_yoy"], True)
    if risk >= 60:
        regime = 50 + 0.32 * (c["momentum"] - 50) + 0.18 * (c["volume"] - 50) + 0.12 * (c["theme"] - 50)
    elif risk <= 42:
        regime = 50 + 0.28 * (c["fundamental"] - 50) - 0.22 * (c["momentum"] - 50) - 0.12 * (beta_rank - 50) + 0.10 * (quality - 50)
    else:
        regime = 50 + 0.10 * (quality - 50) - 0.06 * (c["theme"] - 50) - 0.06 * (beta_rank - 50)
    regime = pd.Series(regime, index=df.index) - 10.0 * ext_off * (beta_rank / 100.0)
    c["regime"] = _clip(regime)
    return c


def _risk_penalty(df: pd.DataFrame, model_key: str) -> Tuple[pd.Series, List[List[str]]]:
    flags: List[List[str]] = []
    points = []
    for _, row in df.iterrows():
        f: List[str] = []
        p = 0.0
        def add(label: str, pts: float) -> None:
            nonlocal p
            f.append(label)
            p += float(pts)
        def val(col: str, default: float = np.nan) -> float:
            return _finite(row.get(col), default)
        if model_key == "potential":
            if val("ret20") > 0.25:
                add("20日涨幅已大", 7)
            if val("ret60") > 0.50:
                add("60日主线已充分演绎", 6)
            if val("turnover_mean20") > 0.12:
                add("换手偏热，潜伏性价比下降", 4)
            if val("limit_up20") >= 2:
                add("近期涨停偏多", 4)
            if val("consecutive_limit_up", 0) >= 2:
                add("已进入连板加速，不属于潜伏", 6)
            if val("maxdd60") < -0.30:
                add("60日回撤较深", 6)
            if val("ret250") < -0.35:
                add("中期趋势偏弱", 4)
            if val("net_profit", 1) <= 0:
                add("尚未盈利", 3)
            pe = val("pe_ttm")
            if np.isfinite(pe) and (pe < 0 or pe > 120):
                add("估值/盈利异常", 3)
            if val("avg_amount20_yi") < 0.60:
                add("流动性偏弱", 6)
            if val("float_mcap_yi") > 500:
                add("大市值，投机弹性受限", 3)
            if val("pe_ttm") > 200:
                add("估值极端", 4)
            if val("profit_yoy") > 500:
                add("利润增速含低基数", 2)
            if abs(val("profit_yoy", 0)) > 300 or abs(val("rev_yoy", 0)) > 200:
                add("财务同比异常，需核查低基数/重组", 2)
            if val("rsi14") < 25:
                add("超卖未稳", 3)
            if val("big_down5") >= 2:
                add("近期连续大跌", 3)
        else:
            if val("rsi14") > 82:
                add("RSI高位过热", 7)
            if val("dist_ma20") > 0.20:
                add("偏离20日线过大", 7)
            if val("ret5") > 0.25:
                add("5日涨幅过大", 6)
            if val("ret20") > 0.40:
                add("20日涨幅进入击鼓传花区", 5)
            if val("ret20") > 0.65:
                add("20日涨幅极端", 5)
            if val("ret60") > 1.0:
                add("60日翻倍，兑现压力大", 5)
            board_n = val("consecutive_limit_up", 0)
            if board_n >= 7:
                add(f"{int(board_n)}连板，末端接力风险", 8)
            elif board_n >= 4:
                add(f"{int(board_n)}连板，只适合观察/轻仓", 3)
            if val("open_count", 0) >= 3:
                add("涨停反复开板，封板质量差", 5)
            if board_n >= 1 and 0 <= val("seal_ratio", 1) < 0.01:
                add("封单/流通市值偏低", 3)
            if val("lhb_net_yi", 0) < -0.2:
                add("龙虎榜净卖出", 3)
            if val("lhb_top_buy_ratio", 0) > 0.5:
                add("龙虎榜买一集中度过高", 2)
            if val("turnover_mean20") > 0.20:
                add("换手过热", 4)
            if val("maxdd60") < -0.35:
                add("近期深跌", 5)
            if val("close_signal") < val("ma20") and val("ma20") < val("ma60"):
                add("趋势破位", 8)
            if val("avg_amount20_yi") < 1.0:
                add("流动性不足", 6)
            if val("big_down5") >= 2:
                add("近期大跌", 4)
            if val("limit_down20") >= 1:
                add("近期跌停", 6)
            if val("near_high60") > 1.08:
                add("加速偏离前高", 4)
            if val("ret60") > 1.5:
                add("60日涨幅极端", 4)
            if val("net_profit", 1) <= 0:
                add("尚未盈利", 1)
            if abs(val("profit_yoy", 0)) > 300 or abs(val("rev_yoy", 0)) > 200:
                add("财务同比异常，需核查低基数/重组", 2)
        f = list(dict.fromkeys(f))
        flags.append(f)
        points.append(min(30.0, p))
    return pd.Series(points, index=df.index, dtype=float), flags


def _stability_and_confidence(
    components: pd.DataFrame,
    weights: pd.Series,
    base_score: pd.Series,
    penalty: pd.Series,
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    comp_cols = [c for c in weights.index if c in components.columns]
    mat = components[comp_cols].fillna(50).to_numpy(dtype=float)
    w0 = weights.reindex(comp_cols).fillna(0).to_numpy(dtype=float)
    base = base_score.to_numpy(dtype=float)
    base_rank = pd.Series(base, index=components.index).rank(pct=True)
    rng = np.random.default_rng(20261019)
    diffs = []
    for seed in range(6):
        jitter = rng.lognormal(mean=0.0, sigma=0.20, size=len(w0))
        jitter = jitter / max(jitter.sum(), 1e-12)
        j_score = mat @ jitter - penalty.to_numpy(dtype=float)
        j_rank = pd.Series(j_score, index=components.index).rank(pct=True)
        diffs.append(float(np.nanmean(np.abs(j_rank - base_rank))) * 100)
    stability = pd.Series(np.clip(100.0 - np.mean(diffs), 0, 100), index=components.index)

    bars = pd.to_numeric(df.get("bars", 100), errors="coerce").fillna(100)
    bars_score = np.clip(bars / 300.0, 0, 1) * 100
    fin_cols = ["net_profit", "roe", "profit_yoy", "rev_yoy", "pe_ttm"]
    fin_pct = df[fin_cols].notna().mean(axis=1) * 100 if all(c in df for c in fin_cols) else pd.Series(50.0, index=df.index)
    theme_ok = np.where(df["theme_fit"].fillna(0) > 0, 100.0, 30.0)
    concept_ok = np.where(df["concept_count"].fillna(0) > 0, 100.0, 30.0)
    data_quality = 0.45 * bars_score + 0.25 * fin_pct + 0.15 * theme_ok + 0.15 * concept_ok
    top_comp = (components >= 65).sum(axis=1) / max(1, len(comp_cols)) * 100
    confidence = 0.35 * data_quality + 0.35 * stability + 0.30 * top_comp - penalty * 0.55
    confidence = pd.Series(np.clip(confidence, 0, 100), index=components.index)
    return stability, confidence, data_quality, top_comp


def score_model(
    features: pd.DataFrame,
    snapshot: pd.DataFrame,
    model_cfg: Dict[str, Any],
    model_key: str,
    settings: ScanSettings,
    macro: Dict[str, Any],
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    """对一个模型打分并排序。"""
    _set_model_config(model_cfg)
    if features.empty:
        return pd.DataFrame()
    df = _model_eligibility(features, snapshot, model_key, settings, signal_date)
    if df.empty:
        return pd.DataFrame()
    weights = pd.Series(model_cfg[model_key]["weights"], dtype=float)
    comps = _components_potential(df, macro) if model_key == "potential" else _components_active(df, macro)
    penalty, flags = _risk_penalty(df, model_key)
    base = (comps.mul(weights.reindex(comps.columns).fillna(0), axis=1)).sum(axis=1) - penalty
    base = base.clip(0, 100)
    stability, confidence, data_quality, agreement = _stability_and_confidence(comps, weights, base, penalty, df)
    df = df.copy()
    for col in comps.columns:
        df[f"c_{col}"] = comps[col]
    df["score"] = base
    df["penalty"] = penalty
    df["risk_flags"] = flags
    df["stability"] = stability
    df["confidence"] = confidence
    df["data_quality"] = data_quality
    df["agreement"] = agreement
    df["risk_level"] = np.select(
        [penalty <= 0, penalty <= 8, penalty <= 16],
        ["中高", "高", "高"],
        default="极高",
    )
    df["confidence_label"] = pd.cut(
        df["confidence"], bins=[-0.1, 45, 65, 100], labels=["低", "中", "较高"], right=True
    ).astype(str)
    df["score_percentile"] = df["score"].rank(pct=True) * 100
    df["signal_date"] = signal_date.strftime("%Y-%m-%d")
    df["amount20_yi"] = df["avg_amount20_yi"]
    df["turnover20_pct"] = df["turnover_mean20"] * 100
    df["theme_label"] = df["theme_names"].map(lambda x: " / ".join(x) if isinstance(x, (list, tuple)) else str(x or ""))
    df["risk_flags_text"] = df["risk_flags"].map(lambda x: "；".join(x) if x else "模型未见硬伤（不代表低风险）")
    df["name"] = df["name"].fillna("").astype(str)
    df["code"] = df.index.astype(str).str.zfill(6)
    df = df.sort_values(["score", "confidence", "amount20_yi"], ascending=[False, False, False])
    df = df.reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def metadata_for_ui(model_cfg: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    m = model_cfg[model_key]
    return {
        "label": m.get("label", model_key),
        "definition": m.get("definition", ""),
        "weights": {COMPONENT_LABELS.get(k, k): v for k, v in m.get("weights", {}).items()},
        "notes": {COMPONENT_LABELS.get(k, k): COMPONENT_NOTES.get(k, "") for k in m.get("weights", {})},
    }
