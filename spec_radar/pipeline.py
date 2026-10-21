# -*- coding: utf-8 -*-
"""扫描编排：数据抓取 → 特征 → 主题/宏观 → 双模型排序。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .altdata import (
    AltDataClient,
    AnnBundle,
    LimitUpBundle,
    LhbBundle,
    apply_announcements,
    market_sentiment,
    merge_announcements,
    merge_lhb,
    merge_limitup,
)
from .config import (
    ScanSettings,
    load_macro_playbook,
    load_model_config,
)
from .datasource import (
    CN_TZ,
    BoardBundle,
    DataError,
    MarketClient,
    SnapshotBundle,
    china_now,
    determine_signal_date,
)
from .forecast import load_params as load_forecast_params, predict_forecasts
from .framework import add_active_framework, add_potential_framework, add_trade_levels, build_daily_screens
from .learning import (
    apply_calibration,
    calibrate as calibrate_learning,
    evaluate_ledger,
    load_calibration,
    record_predictions,
)
from .indicators import compute_feature_frame
from .macro import build_macro_context
from .scoring import (
    base_eligibility,
    prefilter_codes,
    score_model,
)
from .wave import adjust_wave_targets_for_display, build_wave_frame
from .themes import (
    CONCEPT_BLACKLIST,
    ThemeMapper,
    attach_theme_features,
    compute_concept_stats,
    compute_industry_stats,
)

LOG = logging.getLogger(__name__)
Progress = Optional[Callable[[str, float, str], None]]


def _report(progress: Progress, stage: str, frac: float, detail: str = "") -> None:
    LOG.info("%s %.0f%% %s", stage, frac * 100, detail)
    if progress:
        try:
            progress(stage, float(np.clip(frac, 0, 1)), detail)
        except Exception:
            pass


def _fingerprint(settings: ScanSettings, model_cfg: Dict[str, Any]) -> str:
    raw = json.dumps(settings.key(), sort_keys=True, ensure_ascii=False) + str(model_cfg.get("version", ""))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _neutral_macro(snapshot: pd.DataFrame, playbook: Dict[str, Any], risk_mode: str) -> Dict[str, Any]:
    empty = pd.DataFrame(columns=["secid", "date", "close"])
    return build_macro_context(empty, snapshot, playbook, risk_mode)


def run_scan(
    settings: ScanSettings | None = None,
    model_cfg: Dict[str, Any] | None = None,
    playbook: Dict[str, Any] | None = None,
    progress: Progress = None,
    force: bool = False,
) -> Dict[str, Any]:
    """执行一次完整扫描，返回可 pickle 的结果字典。"""
    started = time.time()
    settings = settings or ScanSettings()
    model_cfg = model_cfg or load_model_config()
    playbook = playbook or load_macro_playbook()
    warnings: List[str] = []
    client = MarketClient(workers=int(model_cfg.get("scan", {}).get("max_workers", 20)))

    _report(progress, "日历", 0.02, "确认最近已收盘交易日")
    signal_date, probe = determine_signal_date(client, settings.as_of)
    signal_ts = pd.Timestamp(signal_date)

    _report(progress, "快照", 0.06, "抓取全 A 股行情快照")
    snap_bundle: SnapshotBundle = client.get_snapshot(force=force)
    snapshot = snap_bundle.df
    if snap_bundle.stale:
        warnings.extend(snap_bundle.warnings or ["快照使用旧缓存"])

    _report(progress, "板块", 0.12, "抓取行业与概念板块")
    boards: BoardBundle = client.get_boards(force=force)
    if boards.stale:
        warnings.extend(boards.warnings or ["板块使用旧缓存"])
    mapper = ThemeMapper(playbook)

    _report(progress, "粗筛", 0.18, "用快照缩小需要拉取日 K 的股票池")
    live_snapshot = snap_bundle.fetched_at.astimezone(CN_TZ).date() > signal_date
    codes, pre_stats = prefilter_codes(
        snapshot, settings, model_cfg, mapper, signal_ts, adjust_live_snapshot=live_snapshot
    )
    if not codes:
        raise DataError("基础域为空：检查板块筛选、市值/成交额阈值或数据日期")
    snap_by_code = snapshot.set_index("code")
    valid_codes = [c for c in codes if c in snap_by_code.index]
    requests_map = {str(snap_by_code.loc[c, "secid"]): c for c in valid_codes}
    _report(progress, "日K", 0.24, f"抓取 {len(requests_map)} 只候选股的前复权日 K")

    kline_bundle = client.get_kline_panel(
        requests_map,
        signal_date,
        lmt=int(settings.lookback_bars or 320),
        force=force,
    )
    panel = kline_bundle.frame
    if kline_bundle.stale:
        warnings.append("候选股日 K 使用旧缓存/离线数据，信号可能滞后")
    warnings.extend(kline_bundle.warnings or [])
    if kline_bundle.failed:
        warnings.append(f"{len(kline_bundle.failed)} 只候选股日 K 缺失，已从评分中剔除")
    if kline_bundle.fallback:
        warnings.append(f"{len(kline_bundle.fallback)} 只股票使用腾讯/新浪备用 K 线，成交额/换手可能为估算值")

    _report(progress, "市场环境", 0.60, "抓取指数与全球/风格代理资产")
    try:
        context_bundle = client.get_market_context_panel(signal_date, lmt=320)
        context_panel = context_bundle.frame
        warnings.extend(context_bundle.warnings or [])
        if context_bundle.failed:
            warnings.append(f"市场环境面板缺失 {len(context_bundle.failed)} 个代理资产，按中性处理")
    except Exception as exc:  # noqa: BLE001
        context_panel = pd.DataFrame()
        warnings.append(f"市场环境面板不可用，宏观状态按中性处理：{exc}")
    macro = build_macro_context(context_panel, snapshot, playbook, settings.risk_mode)
    warnings.extend(macro.get("warnings", []))

    benchmark = pd.DataFrame()
    if not context_panel.empty and "secid" in context_panel.columns:
        benchmark = context_panel[context_panel["secid"] == "1.000001"].copy()
    if benchmark.empty:
        benchmark = probe[probe["date"] <= signal_ts].copy()
        if benchmark.empty:
            warnings.append("缺少上证指数基准，Beta 按中性处理")

    _report(progress, "特征", 0.68, "计算波动压缩、吸筹、动量、量能等特征")
    features = compute_feature_frame(panel, benchmark, snapshot, min_bars=60)
    if features.empty:
        raise DataError("没有足够的日 K 计算特征，请稍后重试或减小扫描范围")

    # 短线辅助数据：涨停池（封单/连板）、龙虎榜。
    # 若涨停池日期不是信号日（盘中常见），只用于展示市场温度，绝不并入信号日的打分。
    alt = AltDataClient(workers=min(8, int(model_cfg.get("scan", {}).get("max_workers", 20))))
    limit_bundle = LimitUpBundle(pd.DataFrame(), None)
    lhb_bundle = LhbBundle(pd.DataFrame(), pd.DataFrame())
    sentiment = market_sentiment(limit_bundle, snapshot)
    if str(os.environ.get("SPEC_RADAR_OFFLINE", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        _report(progress, "短线数据", 0.71, "涨停池 / 龙虎榜 / 板块联动")
        try:
            limit_bundle = alt.get_limitup(signal_date)
            if limit_bundle.warning:
                warnings.append(limit_bundle.warning)
            if limit_bundle.qdate == signal_date and not limit_bundle.pool.empty:
                features = merge_limitup(features, limit_bundle)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"涨停池不可用：{exc}")
        try:
            lhb_bundle = alt.get_lhb(signal_date)
            if lhb_bundle.warning:
                warnings.append(lhb_bundle.warning)
            if not lhb_bundle.daily.empty:
                features = merge_lhb(features, lhb_bundle)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"龙虎榜不可用：{exc}")
        sentiment = market_sentiment(limit_bundle, snapshot)
    else:
        warnings.append("离线模式：跳过涨停池与龙虎榜实时接口")

    # 行业/概念指数 K 线：只抓候选股涉及的板块，严格截至信号日。
    _report(progress, "板块", 0.78, "计算行业与概念板块的 20/60 日相对强弱")
    candidate_snap = snapshot[snapshot["code"].isin(valid_codes)]
    ind_names = sorted(set(candidate_snap["industry"].dropna().astype(str)))
    concept_names = set()
    for concepts in candidate_snap["concepts"]:
        for concept in concepts:
            if concept not in CONCEPT_BLACKLIST:
                concept_names.add(concept)
    ind_lookup = {}
    if not boards.industry.empty:
        ind_lookup = {
            str(row["name"]): str(row["board_code"])
            for _, row in boards.industry.iterrows()
            if str(row.get("name", "")).strip() and str(row.get("board_code", "")).strip()
        }
    con_lookup = {}
    if not boards.concept.empty:
        con_lookup = {
            str(row["name"]): str(row["board_code"])
            for _, row in boards.concept.iterrows()
            if str(row.get("name", "")).strip() and str(row.get("board_code", "")).strip()
        }
    board_codes = set()
    for name in ind_names:
        if name in ind_lookup:
            board_codes.add(ind_lookup[name])
    for name in concept_names:
        if name in con_lookup:
            board_codes.add(con_lookup[name])
    board_requests = {f"90.{code}": "" for code in sorted(board_codes)}
    industry_stats = pd.DataFrame()
    concept_stats = pd.DataFrame()
    board_panel_frame = pd.DataFrame()
    if board_requests:
        try:
            board_panel = client.get_kline_panel(board_requests, signal_date, lmt=130, force=False)
            board_panel_frame = board_panel.frame.copy()
            industry_stats = compute_industry_stats(board_panel.frame, boards.industry, signal_ts)
            concept_stats = compute_concept_stats(board_panel.frame, boards.concept, signal_ts)
            if board_panel.failed:
                warnings.append(f"{len(board_panel.failed)} 个行业/概念板块指数缺失，相关热度按中性处理")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"行业/概念指数 K 线不可用，相对强弱按中性处理：{exc}")
    else:
        warnings.append("板块映射覆盖不足，行业/概念相对强弱按中性处理")

    _report(progress, "主题", 0.84, "匹配政策主题与概念热度")
    try:
        features = attach_theme_features(features, snapshot, mapper, industry_stats, concept_stats, macro)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"主题特征附加失败，按中性主题处理：{exc}")
        features["theme_fit"] = 0.0
        features["theme_tilt"] = 0.0
        features["theme_names"] = [() for _ in range(len(features))]
        features["theme_id"] = ""
        features["theme_proxy_ret20"] = 0.0
        features["ind_ret20"] = 0.0
        features["ind_ret60"] = 0.0
        features["ind_vol20"] = np.nan
        features["ind_breadth"] = 0.5
        features["ind_day"] = 0.0
        features["rs20"] = 0.0
        features["rs60"] = 0.0
        features["concept_day"] = 0.0
        features["concept_20d"] = 0.0
        features["concept_60d"] = 0.0
        features["concept_breadth"] = 0.5
        features["concept_leader_pct"] = 0.0
        features["concept_count"] = 0.0

    _report(progress, "三浪", 0.88, "识别一浪上涨/二浪回调/三浪启动结构")
    # 在进入评分前固定 code 列，避免后续 reset_index/merge 改写代码。
    if "code" not in features.columns:
        features["code"] = features.index.astype(str).str.zfill(6)
    else:
        features["code"] = features["code"].astype(str).str.zfill(6)
    features = features.reset_index(drop=True)
    try:
        wave_frame = build_wave_frame(panel)
        if not wave_frame.empty:
            wave_frame.index = wave_frame.index.astype(str).str.zfill(6)
            wave_frame = wave_frame.reset_index()
            wave_frame["code"] = wave_frame["code"].astype(str).str.zfill(6)
            overlap = [c for c in wave_frame.columns if c in features.columns and c != "code"]
            if overlap:
                features = features.drop(columns=overlap)
            features = features.merge(wave_frame, on="code", how="left")
        else:
            warnings.append("三浪结构识别：无足够 K 线样本")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"三浪结构识别失败（不影响双榜）：{exc}")
    for col, default in (
        ("wave_pattern_score", 0.0), ("wave_stage", "非典型"), ("wave_stage_code", "none"),
        ("wave_profit_score", 0.0), ("wave_launch_prob", np.nan),
    ):
        if col not in features:
            features[col] = default
        elif pd.api.types.is_numeric_dtype(features[col]):
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(default)
        else:
            features[col] = features[col].fillna(default)

    _report(progress, "评分", 0.90, "运行潜在投机与活跃投机双模型")
    potential = score_model(features, snapshot, model_cfg, "potential", settings, macro, signal_ts)
    active = score_model(features, snapshot, model_cfg, "active", settings, macro, signal_ts)
    # 第一浪候选不强制通过三浪/六维/六格门槛，保留评分全集用于独立筛选。
    potential_all = potential.copy()
    active_all = active.copy()
    if potential.empty:
        warnings.append("潜在投机模型没有符合条件的股票：可放宽板块/市值筛选或增大扫描深度")
    if active.empty:
        warnings.append("活跃投机模型没有符合条件的股票：可放宽板块/市值筛选或增大扫描深度")
    if len(potential) < min(settings.n_potential * 2, 20):
        warnings.append(f"潜在榜合格样本仅 {len(potential)} 只，排名稳定性有限")
    if len(active) < min(settings.n_active * 2, 20):
        warnings.append(f"活跃榜合格样本仅 {len(active)} 只，排名稳定性有限")

    # 公告红线：只核查两个榜单的前若干名，避免全市场数百次请求。
    ann_bundle = AnnBundle(pd.DataFrame())
    ann_checked = 0
    if str(os.environ.get("SPEC_RADAR_OFFLINE", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        _report(progress, "公告", 0.93, "核查前排名单的澄清/减持/立案等红线")
        try:
            top_codes = set()
            for frame in (potential, active):
                if not frame.empty:
                    n = min(len(frame), max(40, int(settings.n_potential)))
                    top_codes.update(frame.head(n)["code"].astype(str).tolist())
            if top_codes:
                ann_bundle = alt.get_announcements(top_codes, signal_date)
                if ann_bundle.warning:
                    warnings.append(ann_bundle.warning)
                if not ann_bundle.flags.empty:
                    ann_checked = int(ann_bundle.flags["ann_checked"].sum()) if "ann_checked" in ann_bundle.flags else len(ann_bundle.flags)
                    potential, hard_p = apply_announcements(potential, ann_bundle)
                    active, hard_a = apply_announcements(active, ann_bundle)
                    if hard_p or hard_a:
                        warnings.append(f"公告红线剔除：潜在 {len(hard_p)} 只、活跃 {len(hard_a)} 只（已从榜单移除）")
                    potential = potential[~potential.get("ann_hard_excluded", False)].copy()
                    active = active[~active.get("ann_hard_excluded", False)].copy()
                    for frame in (potential, active):
                        if not frame.empty:
                            frame["rank"] = np.arange(1, len(frame) + 1)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"公告核查失败，榜单未做公告红线剔除：{exc}")
    else:
        warnings.append("离线模式：跳过公告红线核查")

    # 自我学习：先用本次历史面板结算旧账本，再校准本批预测。
    # 关闭方式：SPEC_RADAR_AUTO_LEARNING=0。
    auto_learning = str(os.environ.get("SPEC_RADAR_AUTO_LEARNING", "1")).strip().lower() not in {"0", "false", "no", "off"}
    learning_eval = {"evaluated": 0, "pending": 0}
    learning_cal = load_calibration()
    if auto_learning:
        try:
            learning_eval = evaluate_ledger(panel)
            if learning_eval.get("evaluated", 0) > 0:
                learning_cal = calibrate_learning()
                warnings.append(f"自我学习：结算 {learning_eval['evaluated']} 条历史预测并更新概率/分数校准")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"自我学习评估失败（不影响选股）：{exc}")

    _report(progress, "框架/预测", 0.96, "六维检查、概率、目标价与起爆窗口")
    filtered = {"potential": 0, "active": 0}
    if not potential.empty:
        potential = add_potential_framework(potential)
        potential["score"] = (potential["score"] + 1.5 * (potential["setup_score"] - 8)).clip(0, 100)
        before = len(potential)
        potential = potential[potential["setup_score"] >= 9].copy()
        filtered["potential"] = before - len(potential)
        potential = potential.sort_values(["score", "confidence", "amount20_yi"], ascending=[False, False, False]).reset_index(drop=True)
        potential["rank"] = np.arange(1, len(potential) + 1)
    if not active.empty:
        active = add_active_framework(active)
        active["score"] = (active["score"] + 0.8 * (active["check_count"] - 5)).clip(0, 100)
        before = len(active)
        active = active[(active["check_count"] >= 5) & active["check_safety"]].copy()
        filtered["active"] = before - len(active)
        active = active.sort_values(["score", "confidence", "amount20_yi"], ascending=[False, False, False]).reset_index(drop=True)
        active["rank"] = np.arange(1, len(active) + 1)
    if filtered["potential"] or filtered["active"]:
        warnings.append(
            f"框架门槛剔除：潜在 {filtered['potential']} 只（六维<9分）、"
            f"活跃 {filtered['active']} 只（六格<5项或安全红线）"
        )
    forecast_params = load_forecast_params()
    if not potential.empty:
        potential = predict_forecasts(potential, "potential", forecast_params)
    if not active.empty:
        active = predict_forecasts(active, "active", forecast_params)
    if not potential.empty:
        potential = apply_calibration(potential, "potential", learning_cal)
    if not active.empty:
        active = apply_calibration(active, "active", learning_cal)
    if not potential.empty:
        potential = add_trade_levels(potential, "potential")
    if not active.empty:
        active = add_trade_levels(active, "active")
    # “盈利能力”在用户语境里是未来股价可能拉升的总幅度，不是公司财务盈利。
    # 因此默认按预测目标空间排序；财务质量只作为参考列，不再作为默认排序键。
    def _add_rally_space(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        cur = pd.to_numeric(
            out.get("current_price", out.get("close_signal")), errors="coerce"
        ).replace(0, np.nan)
        cur = cur.fillna(pd.to_numeric(out.get("close_signal"), errors="coerce"))
        base = (pd.to_numeric(out.get("target_high"), errors="coerce") / cur - 1.0).fillna(0).clip(lower=0)
        bull = (pd.to_numeric(out.get("target_high_bull"), errors="coerce") / cur - 1.0).fillna(0).clip(lower=0)
        if "wave_target2" in out:
            wave_bull = (pd.to_numeric(out["wave_target2"], errors="coerce") / cur - 1.0).fillna(0).clip(lower=0)
            bull = np.maximum(bull, wave_bull)
        out["rally_space_pct"] = base
        out["rally_max_pct"] = bull
        out["rally_score"] = (base * 100.0).clip(0, 100)
        if "first_wave_pattern_score" in out:
            out["first_wave_composite"] = (
                0.35 * pd.to_numeric(out["first_wave_pattern_score"], errors="coerce").fillna(0)
                + 0.25 * pd.to_numeric(out.get("first_wave_launch_prob"), errors="coerce").fillna(0) * 100
                + 0.15 * pd.to_numeric(out.get("first_wave_upside_bull"), errors="coerce").fillna(0).clip(0, 0.6) * 100
                + 0.10 * pd.to_numeric(out.get("profitability_score"), errors="coerce").fillna(0)
                + 0.08 * pd.to_numeric(out.get("theme_fit"), errors="coerce").fillna(0) * 100
                + 0.07 * pd.to_numeric(out.get("score"), errors="coerce").fillna(0)
            ).clip(0, 100)
        return out

    potential = _add_rally_space(potential)
    active = _add_rally_space(active)

    def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["model_rank"] = out["rank"]
        mode = "fundamental" if settings.sort_mode == "profitability" else settings.sort_mode
        if mode == "rally":
            out = out.sort_values(
                ["rally_space_pct", "p_hit_10d", "rally_max_pct", "score", "confidence", "amount20_yi"],
                ascending=[False, False, False, False, False, False],
            ).reset_index(drop=True)
        elif mode == "rally_max":
            out = out.sort_values(
                ["rally_max_pct", "p_hit_10d", "score", "confidence", "amount20_yi"],
                ascending=[False, False, False, False, False],
            ).reset_index(drop=True)
        elif mode == "fundamental":
            out = out.sort_values(
                ["profitability_score", "score", "confidence", "amount20_yi"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
        else:
            out = out.sort_values(
                ["score", "confidence", "amount20_yi"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    potential = adjust_wave_targets_for_display(_sort_frame(potential))
    active = adjust_wave_targets_for_display(_sort_frame(active))
    # 三浪雷达：从两个榜单的合格候选中再按“结构+赔率+概率+盈利/主题”排序。
    wave_lists = {"typical": pd.DataFrame(), "pre": pd.DataFrame(), "first": pd.DataFrame()}
    wave_parts = []
    for label, frame in (("潜在", potential), ("活跃", active)):
        if not frame.empty and "wave_stage_code" in frame:
            part = frame.copy()
            part["wave_model"] = label
            wave_parts.append(part)
    if wave_parts:
        wave_all = pd.concat(wave_parts, ignore_index=True)
        if "wave_stage_code" in wave_all:
            def _numcol(name: str, default: float = 0.0) -> pd.Series:
                if name not in wave_all:
                    return pd.Series(default, index=wave_all.index, dtype=float)
                return pd.to_numeric(wave_all[name], errors="coerce").fillna(default)

            wave_all["wave_composite"] = (
                0.32 * _numcol("wave_pattern_score")
                + 0.20 * _numcol("wave_launch_prob") * 100
                + 0.14 * _numcol("wave_upside_bull").clip(0, 0.6) * 100
                + 0.10 * _numcol("rally_space_pct").clip(0, 1.0) * 100
                + 0.09 * _numcol("theme_fit") * 100
                + 0.15 * _numcol("score")
            ).clip(0, 100)
            wave_all = wave_all.sort_values(["wave_composite", "rally_max_pct"], ascending=[False, False])
            wave_all = wave_all.drop_duplicates("code", keep="first").reset_index(drop=True)
            typical = wave_all[
                wave_all["wave_stage_code"].isin(["early3", "main3"])
                & (_numcol("wave_pattern_score") >= 60)
                & (_numcol("wave_upside_bull") >= 0.05)
                & (wave_all["wave_composite"] >= 50)
            ].copy()
            pre = wave_all[
                wave_all["wave_stage_code"].eq("pre3")
                & (_numcol("wave_pattern_score") >= 60)
                & (_numcol("wave_launch_prob") >= 0.35)
                & (wave_all["wave_composite"] >= 50)
            ].copy()
            for frame in (typical, pre):
                if not frame.empty:
                    # 信号日目标若已被盘中现价越过，展示上方下一目标，避免“目标价低于现价”。
                    cur = pd.to_numeric(frame.get("current_price"), errors="coerce")
                    t1 = pd.to_numeric(frame.get("wave_target1"), errors="coerce")
                    t2 = pd.to_numeric(frame.get("wave_target2"), errors="coerce")
                    frame["wave_target1"] = np.where(cur.notna() & t1.notna() & (cur > t1), cur * 1.03, t1)
                    frame["wave_target2"] = np.maximum(
                        np.where(cur.notna() & t2.notna() & (cur > t2), cur * 1.06, t2),
                        pd.to_numeric(frame["wave_target1"], errors="coerce") * 1.03,
                    )
                    frame["wave_sell_low"] = np.maximum(
                        pd.to_numeric(frame.get("wave_sell_low"), errors="coerce"),
                        pd.to_numeric(frame["wave_target1"], errors="coerce"),
                    )
                    frame["wave_sell_high"] = np.maximum(
                        pd.to_numeric(frame.get("wave_sell_high"), errors="coerce"),
                        pd.to_numeric(frame["wave_target2"], errors="coerce"),
                    )
                    frame["wave_upside_base"] = (
                        pd.to_numeric(frame["wave_target1"], errors="coerce") / cur - 1.0
                    ).clip(lower=0)
                    frame["wave_upside_bull"] = (
                        pd.to_numeric(frame["wave_target2"], errors="coerce") / cur - 1.0
                    ).clip(lower=0)
                    frame["wave_rank"] = np.arange(1, len(frame) + 1)

            # 第一浪启动前：平台收敛/低点抬高/缩量，且尚未进入已完成的一浪结构。
            # 第一浪与三浪可能是不同级别的结构，允许重叠；这里只按第一浪自身条件筛。
            first_parts = []
            for label, base_frame in (("潜在", potential_all), ("活跃", active_all)):
                if not base_frame.empty and "first_wave_stage_code" in base_frame:
                    part = base_frame.copy()
                    part["wave_model"] = label
                    first_parts.append(part)
            first = pd.DataFrame()
            if first_parts:
                all_first = pd.concat(first_parts, ignore_index=True)
                all_first = all_first[
                    all_first["first_wave_stage_code"].isin(["pre1", "pre1_break"])
                    & (pd.to_numeric(all_first["first_wave_pattern_score"], errors="coerce").fillna(0) >= 55)
                    & (pd.to_numeric(all_first["first_wave_launch_prob"], errors="coerce").fillna(0) >= 0.30)
                ].copy()
                if not all_first.empty:
                    def _fcol(name: str, default: float = 0.0) -> pd.Series:
                        if name not in all_first:
                            return pd.Series(default, index=all_first.index, dtype=float)
                        return pd.to_numeric(all_first[name], errors="coerce").fillna(default)

                    all_first["first_wave_composite"] = (
                        0.35 * _fcol("first_wave_pattern_score")
                        + 0.25 * _fcol("first_wave_launch_prob") * 100
                        + 0.15 * _fcol("first_wave_upside_bull").clip(0, 0.6) * 100
                        + 0.10 * _fcol("profitability_score")
                        + 0.08 * _fcol("theme_fit") * 100
                        + 0.07 * _fcol("score")
                    ).clip(0, 100)
                    first = all_first[all_first["first_wave_composite"] >= 45].copy()
                    first = first.sort_values(["first_wave_composite", "first_wave_upside_bull"], ascending=[False, False])
                    first = first.drop_duplicates("code", keep="first").reset_index(drop=True)
                    first["first_wave_rank"] = np.arange(1, len(first) + 1)
            wave_lists = {"typical": typical, "pre": pre, "first": first}
            composite_map = wave_all[["code", "wave_composite"]].drop_duplicates("code")
            if not potential.empty and "wave_composite" not in potential:
                potential = potential.merge(composite_map, on="code", how="left")
            if not active.empty and "wave_composite" not in active:
                active = active.merge(composite_map, on="code", how="left")
    screens = build_daily_screens(features, snapshot, {"potential": potential, "active": active})

    elapsed = time.time() - started
    # 把离线/缓存降级合并成一条，不要把相同提示重复五遍。
    detail_warnings = list(dict.fromkeys(warnings))
    offline_like = [w for w in detail_warnings if any(x in w for x in ("离线模式", "离线数据", "内置离线", "旧缓存", "不是实时行情"))]
    if offline_like:
        detail_warnings = [w for w in detail_warnings if w not in offline_like]
        detail_warnings.append(
            "数据源为内置离线灾备/旧缓存：不是实时行情；信号日、快照时间与来源见页面顶部。"
        )
    index_missing = [w.replace("指数缺失：", "") for w in detail_warnings if "指数缺失" in w]
    board_missing = [w for w in detail_warnings if "板块" in w and ("不可用" in w or "缺失" in w)]
    offline_mode = bool(offline_like)
    data_health = {
        "mode": "offline_bootstrap" if offline_mode else ("cache" if snap_bundle.stale or kline_bundle.stale else "live"),
        "offline": offline_mode,
        "board_ok": bool(boards is not None and not boards.industry.empty and not boards.concept.empty and not industry_stats.empty),
        "index_missing": index_missing,
        "board_missing": board_missing,
        "limitup_ok": bool(limit_bundle.qdate == signal_date and not limit_bundle.pool.empty),
        "ann_checked": int(ann_checked),
        "stale": bool(snap_bundle.stale or kline_bundle.stale or boards.stale),
    }
    meta = {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "snapshot_time": snap_bundle.fetched_at.astimezone(CN_TZ).strftime("%Y-%m-%d %H:%M"),
        "generated_at": china_now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
        "universe_total": int(len(snapshot)),
        "eligible_base": int(pre_stats.get("base", 0)),
        "preselected": int(len(valid_codes)),
        "features": int(len(features)),
        "potential_scored": int(len(potential)),
        "active_scored": int(len(active)),
        "warnings": detail_warnings,
        "warnings_detail": list(dict.fromkeys(warnings)),
        "data_health": data_health,
        "stale": bool(snap_bundle.stale or kline_bundle.stale or boards.stale),
        "model_version": str(model_cfg.get("version", "")),
        "config_fingerprint": _fingerprint(settings, model_cfg),
        "macro_label": str(macro.get("label", "")),
        "risk_appetite": round(float(macro.get("risk_appetite", 50)), 1),
        "kline_rows": int(len(panel)),
        "kline_missing": int(len(kline_bundle.failed)),
        "kline_fallback": int(len(kline_bundle.fallback)),
        "source": (
            "内置离线数据（灾备）" if any("内置离线" in w for w in warnings)
            else "东方财富公开行情 + 腾讯/新浪备用 K 线"
        ),
        "cache_dir": str(client.cache.directory),
        "sentiment": sentiment,
        "limitup_rows": int(len(limit_bundle.pool)) if not limit_bundle.pool.empty else 0,
        "lhb_rows": int(len(lhb_bundle.daily)) if not lhb_bundle.daily.empty else 0,
        "ann_checked": int(ann_checked),
        "forecast_calibrated": bool((forecast_params.get("models") or {}).get("potential", {}).get("ok")),
        "forecast_auc_potential": (forecast_params.get("models") or {}).get("potential", {}).get("auc_hit"),
        "forecast_auc_active": (forecast_params.get("models") or {}).get("active", {}).get("auc_hit"),
        "learning_calibrated": bool((learning_cal.get("models") or {}).get("potential", {}).get("ok")),
        "learning_evaluated": int(learning_eval.get("evaluated", 0)),
        "learning_pending": int(learning_eval.get("pending", 0)),
        "sort_mode": settings.sort_mode,
        "wave_typical": int(len(wave_lists.get("typical", []))),
        "wave_pre": int(len(wave_lists.get("pre", []))),
        "wave_first": int(len(wave_lists.get("first", []))),
    }
    if auto_learning:
        try:
            n_learn = min(20, max(5, int(settings.n_potential)))
            recorded = record_predictions(
                {"meta": meta, "potential": potential, "active": active},
                n_learn, n_learn,
            )
            meta["learning_recorded"] = int(recorded)
        except Exception as exc:  # noqa: BLE001
            meta["learning_recorded"] = 0
            warnings.append(f"学习账本记录失败（不影响选股）：{exc}")
    else:
        meta["learning_recorded"] = 0
    _report(progress, "完成", 1.0, f"潜在 {len(potential)} 只 / 活跃 {len(active)} 只")
    return {
        "potential": potential,
        "active": active,
        "kline": panel,
        "board_panel": board_panel_frame,
        "macro": macro,
        "meta": meta,
        "industry_stats": industry_stats,
        "limitup": limit_bundle,
        "lhb": lhb_bundle,
        "sentiment": sentiment,
        "announcements": ann_bundle,
        "screens": screens,
        "wave_lists": wave_lists,
        "forecast_params": forecast_params,
        "learning_calibration": learning_cal,
    }
