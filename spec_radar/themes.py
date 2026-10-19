# -*- coding: utf-8 -*-
"""行业/概念映射、主题政策契合度与行业相对强弱。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 不是“产业主题”的宽基/风格/交易标签，避免给股票虚增题材分。
CONCEPT_BLACKLIST = {
    "融资融券", "深股通", "沪股通", "标普道琼斯A股", "MSCI中国", "富时罗素",
    "机构重仓", "基金重仓", "社保重仓", "QFII重仓", "预盈预增", "预亏预减",
    "年报预增", "中报预增", "高送转", "破净股", "低价股", "微盘股", "次新股",
    "创业板综", "深成500", "上证380", "中证500", "沪深300", "上证50",
    "中证1000", "百元股", "昨日涨停", "昨日连板", "昨日触板", "转债标的",
    "科创板做市商", "科创板做市股", "IPO受益", "独角兽", "创投",
    "参股银行", "参股保险", "参股券商", "参股期货", "深圳特区", "滨海新区",
    "长江三角", "西部大开发", "东北振兴", "京津冀", "雄安新区", "海南自贸",
    "粤港自贸", "上海自贸", "成渝特区", "湖北自贸", "沪企改革", "央国企改革",
    "并购重组概念", "化债(AMC)概念", "REITs概念", "PPP模式", "一带一路",
}


@dataclass
class ThemeMatch:
    fit: float = 0.0
    tilt: float = 0.0
    names: Tuple[str, ...] = ()
    dominant_id: str = ""


class ThemeMapper:
    """把人工主题底稿转换成每只股票的政策/产业催化暴露。"""

    def __init__(self, playbook: Dict[str, Any]) -> None:
        self.themes = list(playbook.get("themes", []))
        self.by_id = {str(t.get("id")): t for t in self.themes}
        self.industry_map: Dict[str, List[Tuple[str, float]]] = {}
        self.keyword_exact: Dict[str, List[Tuple[str, float]]] = {}
        self.all_keywords: List[Tuple[str, str, float]] = []
        for theme in self.themes:
            tid = str(theme.get("id"))
            for industry in theme.get("industries", []) or []:
                self.industry_map.setdefault(str(industry), []).append((tid, 0.72))
            for keyword in theme.get("keywords", []) or []:
                kw = str(keyword).strip()
                if not kw:
                    continue
                self.keyword_exact.setdefault(kw, []).append((tid, 1.0))
                self.all_keywords.append((kw, tid, 0.82))

    def match(self, industry: str, concepts: Sequence[str]) -> ThemeMatch:
        evidence: Dict[str, float] = {}
        for tid, weight in self.industry_map.get(str(industry), []):
            evidence[tid] = max(evidence.get(tid, 0.0), weight)
        concept_set = [str(c).strip() for c in concepts if str(c).strip()]
        for concept in concept_set:
            for tid, weight in self.keyword_exact.get(concept, []):
                evidence[tid] = max(evidence.get(tid, 0.0), weight)
            # 子串匹配只对较长的关键词生效，避免 "AI" 这类过泛匹配。
            for kw, tid, weight in self.all_keywords:
                if len(kw) >= 3 and kw != concept and kw in concept:
                    evidence[tid] = max(evidence.get(tid, 0.0), weight)
        if not evidence:
            return ThemeMatch()
        ranked = sorted(evidence.items(), key=lambda x: x[1], reverse=True)
        top_id, top_weight = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        fit = float(np.clip(top_weight + 0.12 * second, 0.0, 1.0))
        tilt = float(self.by_id.get(top_id, {}).get("tilt", 0.0))
        names = tuple(
            str(self.by_id.get(tid, {}).get("name", tid))
            for tid, _ in ranked[:2]
        )
        return ThemeMatch(fit=fit, tilt=tilt, names=names, dominant_id=top_id)


def _board_kline_stats(
    board_panel: pd.DataFrame,
    boards: pd.DataFrame,
    name_col: str,
    asof: pd.Timestamp | None = None,
    prefix: str = "ind",
) -> pd.DataFrame:
    """把板块指数 K 线转换为信号日截面统计，严格截至 signal_date。"""
    out_cols = [name_col, f"{prefix}_ret20", f"{prefix}_ret60", f"{prefix}_vol20", f"{prefix}_amount_ratio", f"{prefix}_day"]
    if board_panel is None or board_panel.empty or boards is None or boards.empty:
        return pd.DataFrame(columns=out_cols)
    lookup = {
        str(row["board_code"]): str(row["name"])
        for _, row in boards.iterrows()
        if str(row.get("board_code", "")).strip()
    }
    rows = []
    panel = board_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if asof is not None:
        panel = panel[panel["date"] <= pd.Timestamp(asof)]
    for secid, group in panel.groupby("secid", sort=False):
        code = str(secid).split(".")[-1]
        name = lookup.get(code) or lookup.get(str(secid))
        if not name:
            continue
        g = group.sort_values("date").drop_duplicates("date", keep="last")
        close = g["close"].to_numpy(dtype=float)
        if len(close) < 5:
            continue
        rets = np.diff(close) / np.maximum(close[:-1], 1e-12)
        amount = g["amount"].to_numpy(dtype=float) if "amount" in g else np.array([])
        rows.append({
            name_col: name,
            f"{prefix}_ret20": float(close[-1] / close[-21] - 1) if len(close) > 20 else np.nan,
            f"{prefix}_ret60": float(close[-1] / close[-61] - 1) if len(close) > 60 else np.nan,
            f"{prefix}_vol20": float(np.std(rets[-20:], ddof=1) * np.sqrt(252)) if len(rets) >= 20 else np.nan,
            f"{prefix}_amount_ratio": float(np.mean(amount[-5:]) / max(np.mean(amount[-20:]), 1e-12)) if len(amount) >= 20 else np.nan,
            f"{prefix}_day": float(close[-1] / close[-2] - 1) if len(close) > 1 else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows).drop_duplicates(name_col, keep="first")


def compute_industry_stats(
    board_panel: pd.DataFrame,
    industry_boards: pd.DataFrame,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """用行业指数 K 线计算信号日的行业动量/波动/量能，不使用盘中板块快照。"""
    result = _board_kline_stats(board_panel, industry_boards, "industry", asof, prefix="ind")
    if result.empty:
        return result
    # 宽度将在 attach_theme_features 中用候选股截面补，不采用实时快照。
    result["ind_breadth"] = 0.5
    return result


def compute_concept_stats(
    board_panel: pd.DataFrame,
    concept_boards: pd.DataFrame,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    result = _board_kline_stats(board_panel, concept_boards, "concept", asof, prefix="concept")
    return result


def attach_theme_features(
    features: pd.DataFrame,
    snapshot: pd.DataFrame,
    mapper: ThemeMapper,
    industry_stats: pd.DataFrame,
    concept_stats: pd.DataFrame | None,
    macro: Dict[str, Any],
) -> pd.DataFrame:
    """在特征表上增加行业/概念相对强度、主题契合度；全部截至信号日。"""
    if features.empty:
        return features
    df = features.copy()
    snap = snapshot.set_index("code")
    industry_lookup = industry_stats.set_index("industry").to_dict("index") if not industry_stats.empty else {}

    # 行业宽度用候选股信号日截面近似，避免使用盘中快照的涨跌家数。
    industry_breadth: Dict[str, float] = {}
    if not snapshot.empty:
        code_meta = snapshot.set_index("code").reindex(df.index.astype(str))
        tmp = pd.DataFrame({"industry": code_meta["industry"].values, "ret60": df["ret60"].values}, index=df.index)
        industry_breadth = tmp.groupby("industry")["ret60"].apply(lambda s: float((s > 0).mean())).to_dict()

    concept_lookup: Dict[str, Dict[str, float]] = {}
    if concept_stats is not None and not concept_stats.empty:
        for _, row in concept_stats.iterrows():
            name = str(row.get("concept", "")).strip()
            if not name:
                continue
            concept_lookup[name] = {
                "pct": float(row.get("concept_day", np.nan)),
                "ret20": float(row.get("concept_ret20", np.nan)),
                "ret60": float(row.get("concept_ret60", np.nan)),
            }
    theme_returns = macro.get("theme_returns", {}) or {}
    rows = []
    for code, feat in df.iterrows():
        code = str(code).zfill(6)
        info = snap.loc[code] if code in snap.index else None
        industry = str(info["industry"]) if info is not None else "未分类"
        concepts = list(info["concepts"]) if info is not None else []
        ind = industry_lookup.get(industry, {})
        ind20 = float(ind.get("ind_ret20", 0.0)) if np.isfinite(ind.get("ind_ret20", np.nan)) else 0.0
        ind60 = float(ind.get("ind_ret60", 0.0)) if np.isfinite(ind.get("ind_ret60", np.nan)) else 0.0
        ind_vol = float(ind.get("ind_vol20", np.nan)) if np.isfinite(ind.get("ind_vol20", np.nan)) else np.nan
        ind_breadth = float(industry_breadth.get(industry, 0.5))
        ind_day = float(ind.get("ind_day", 0.0)) if np.isfinite(ind.get("ind_day", np.nan)) else 0.0
        concept_rows = []
        for concept in concepts:
            if concept in CONCEPT_BLACKLIST:
                continue
            row = concept_lookup.get(concept)
            if row:
                concept_rows.append(row)
        if concept_rows:
            cr = pd.DataFrame(concept_rows).sort_values("pct", ascending=False).head(3)
            c_day = float(cr["pct"].mean())
            c_20 = float(cr["ret20"].mean())
            c_60 = float(cr["ret60"].mean())
        else:
            c_day = c_60 = c_20 = 0.0
        match = mapper.match(industry, concepts)
        theme_ret = float(theme_returns.get(match.dominant_id, 0.0) or 0.0)
        rows.append({
            "industry": industry,
            "theme_fit": match.fit,
            "theme_tilt": match.tilt,
            "theme_names": match.names,
            "theme_id": match.dominant_id,
            "theme_proxy_ret20": theme_ret,
            "ind_ret20": ind20,
            "ind_ret60": ind60,
            "ind_vol20": ind_vol,
            "ind_breadth": ind_breadth,
            "ind_day": ind_day,
            "rs20": float(feat.get("ret20", np.nan) - ind20),
            "rs60": float(feat.get("ret60", np.nan) - ind60),
            "concept_day": c_day,
            "concept_20d": c_20,
            "concept_60d": c_60,
            "concept_breadth": 0.5,
            "concept_leader_pct": 0.0,
            "concept_count": float(len(concept_rows)),
        })
    add = pd.DataFrame(rows, index=df.index)
    for col in add.columns:
        df[col] = add[col]
    return df
