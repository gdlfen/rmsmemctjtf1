# -*- coding: utf-8 -*-
"""短线投机辅助数据：涨停池、龙虎榜、公告风险。

这些接口不是行情主链路：失败时返回空表并降级，绝不阻塞选股。
- 涨停池：东财 push2ex，含连板数、封单额、开板次数、行业；
- 龙虎榜：东财 datacenter，含净买入、席位集中度；
- 公告：东财公告接口，关键词识别澄清/减持/立案/问询等红线。
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .cache import DiskCache
from .config import CACHE_DIR
from .datasource import CN_TZ, UA, china_now

ZT_URL = "https://push2ex.eastmoney.com/getTopicZTPool"
LHB_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
ANN_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
UT = "7eea3edcaed734bea9cbfc24409ed989"

ANN_RED_PATTERNS = [
    ("澄清/无关联", r"澄清|无相关业务|无业务关联|不存在.*业务|未涉及|无关联"),
    ("立案/调查", r"立案|调查|处罚|违法违规"),
    ("退市风险", r"退市风险|终止上市|暂停上市"),
    ("减持计划", r"减持计划|拟减持|减持股份"),
    ("问询/监管", r"问询|监管工作函|关注函|警示函"),
    ("业绩暴雷", r"预亏|巨亏|业绩大幅下滑|商誉减值"),
]
ANN_GOOD_PATTERNS = [
    ("重组/注入", r"重组|资产注入|收购|控制权变更|股权转让"),
    ("订单/中标", r"中标|订单|签订|合同"),
    ("业绩改善", r"预增|扭亏|业绩增长|净利润增长"),
    ("技术/产品", r"研发|量产|投产|新品|专利|突破"),
]


@dataclass
class LimitUpBundle:
    pool: pd.DataFrame
    qdate: Optional[date]
    warning: str = ""
    stale: bool = False


@dataclass
class LhbBundle:
    daily: pd.DataFrame
    seats: pd.DataFrame
    warning: str = ""
    stale: bool = False


@dataclass
class AnnBundle:
    flags: pd.DataFrame
    warning: str = ""
    stale: bool = False


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _date_str(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value[:10]
    return value.strftime("%Y-%m-%d")


class AltDataClient:
    def __init__(self, cache: DiskCache | None = None, workers: int = 8, timeout: float = 12.0) -> None:
        self.cache = cache or DiskCache(CACHE_DIR)
        self.workers = workers
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------- 涨停池
    def get_limitup(self, signal_date: date, force: bool = False) -> LimitUpBundle:
        key = f"limitup_{signal_date:%Y%m%d}"
        cached = None if force else self.cache.get(key, max_age_sec=1800)
        if cached is not None:
            return cached.value
        params = {
            "ut": UT, "dpt": "wz.ztzt", "Pageindex": 0, "pagesize": 600,
            "sort": "fbt:asc", "date": signal_date.strftime("%Y%m%d"),
            "_": int(time.time() * 1000),
        }
        try:
            data = self._get(ZT_URL, params)
        except Exception as exc:  # noqa: BLE001
            stale = self.cache.get(key, max_age_sec=None)
            if stale:
                bundle = stale.value
                bundle.stale = True
                return bundle
            return LimitUpBundle(pd.DataFrame(), None, warning=f"涨停池接口失败：{str(exc)[:100]}")
        node = data.get("data") or {}
        qdate_raw = node.get("qdate")
        try:
            qdate = datetime.strptime(str(qdate_raw), "%Y%m%d").date() if qdate_raw else None
        except ValueError:
            qdate = None
        if qdate and qdate != signal_date:
            # 该接口只提供当前/最近交易日，盘中用它判断历史信号会引入未来数据。
            bundle = LimitUpBundle(
                pd.DataFrame(), qdate,
                warning=f"涨停池日期为 {qdate}，不是信号日 {signal_date}，本次不并入模型",
            )
            self.cache.set(key, bundle, {"qdate": str(qdate), "rows": 0})
            return bundle
        rows = node.get("pool") or []
        if not rows:
            return LimitUpBundle(pd.DataFrame(), qdate, warning="涨停池为空")
        df = pd.DataFrame(rows)
        rename = {
            "c": "code", "n": "name", "p": "price_x100", "zdp": "pct",
            "amount": "amount", "ltsz": "float_mcap", "tshare": "total_mcap",
            "hs": "turnover", "lbc": "board_count", "fbt": "first_time",
            "lbt": "last_time", "fund": "seal_amount", "zbc": "open_count",
            "hybk": "industry",
        }
        df = df.rename(columns=rename)
        df["code"] = df["code"].astype(str).str.zfill(6)
        for col in df.columns:
            if col not in {"code", "name", "industry"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["price"] = df["price_x100"] / 100.0
        df["board_count"] = df["board_count"].fillna(0).astype(int)
        df["seal_amount_yi"] = df["seal_amount"] / 1e8
        df["seal_ratio"] = df["seal_amount"] / df["float_mcap"].clip(lower=1)
        df["float_mcap_yi"] = df["float_mcap"] / 1e8
        df["turnover"] = df["turnover"] / 100.0
        df["pct"] = df["pct"] / 100.0
        df["open_count"] = df["open_count"].fillna(0).astype(int)
        df["first_time"] = df["first_time"].fillna(0).astype(int)
        df["last_time"] = df["last_time"].fillna(0).astype(int)
        if "zttj" in df:
            df["days"] = df["zttj"].map(lambda x: (x or {}).get("days", np.nan) if isinstance(x, dict) else np.nan)
        sector_counts = df.groupby("industry")["code"].count().rename("sector_limitup_count")
        df = df.merge(sector_counts, left_on="industry", right_index=True, how="left")
        keep = ["code", "name", "price", "pct", "amount", "float_mcap_yi", "turnover",
                "board_count", "first_time", "last_time", "seal_amount_yi", "seal_ratio",
                "open_count", "industry", "sector_limitup_count"]
        df = df[[c for c in keep if c in df.columns]]
        bundle = LimitUpBundle(df, qdate)
        self.cache.set(key, bundle, {"qdate": str(qdate), "rows": int(len(df))})
        return bundle

    # ------------------------------------------------------------- 龙虎榜
    def get_lhb(self, signal_date: date, force: bool = False) -> LhbBundle:
        key = f"lhb_{signal_date:%Y%m%d}"
        cached = None if force else self.cache.get(key, max_age_sec=6 * 3600)
        if cached is not None:
            return cached.value
        day = _date_str(signal_date)
        daily_params = {
            "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW", "columns": "ALL",
            "filter": f"(TRADE_DATE='{day}')", "pageNumber": 1, "pageSize": 800,
            "sortTypes": "-1", "sortColumns": "BILLBOARD_NET_AMT",
            "source": "WEB", "client": "WEB",
        }
        seat_params = {
            "reportName": "RPT_BILLBOARD_DAILYDETAILSBUY", "columns": "ALL",
            "filter": f"(TRADE_DATE='{day}')", "pageNumber": 1, "pageSize": 2000,
            "source": "WEB", "client": "WEB",
        }
        try:
            daily_json = self._get(LHB_URL, daily_params)
            seat_json = self._get(LHB_URL, seat_params)
        except Exception as exc:  # noqa: BLE001
            stale = self.cache.get(key, max_age_sec=None)
            if stale:
                bundle = stale.value
                bundle.stale = True
                return bundle
            return LhbBundle(pd.DataFrame(), pd.DataFrame(), warning=f"龙虎榜接口失败：{str(exc)[:100]}")
        daily_rows = ((daily_json.get("result") or {}).get("data") or [])
        seat_rows = ((seat_json.get("result") or {}).get("data") or [])
        daily = pd.DataFrame(daily_rows)
        seats = pd.DataFrame(seat_rows)
        if daily.empty:
            return LhbBundle(pd.DataFrame(), pd.DataFrame(), warning="龙虎榜当日无数据")
        daily = daily.rename(columns={
            "SECURITY_CODE": "code", "SECURITY_NAME_ABBR": "name",
            "BILLBOARD_NET_AMT": "lhb_net", "BILLBOARD_BUY_AMT": "lhb_buy",
            "BILLBOARD_SELL_AMT": "lhb_sell", "BILLBOARD_DEAL_AMT": "lhb_deal",
            "DEAL_NET_RATIO": "lhb_net_ratio", "TURNOVERRATE": "lhb_turnover",
            "EXPLANATION": "lhb_reason", "CHANGE_RATE": "lhb_pct",
            "BUY_RATIO": "lhb_buy_ratio", "SELL_RATIO": "lhb_sell_ratio",
        })
        daily["code"] = daily["code"].astype(str).str.zfill(6)
        for col in ("lhb_net", "lhb_buy", "lhb_sell", "lhb_deal", "lhb_net_ratio", "lhb_buy_ratio", "lhb_sell_ratio"):
            if col in daily:
                daily[col] = pd.to_numeric(daily[col], errors="coerce")
        if not seats.empty:
            seats = seats.rename(columns={
                "SECURITY_CODE": "code", "OPERATEDEPT_NAME": "seat_name",
                "BUY": "seat_buy", "SELL": "seat_sell", "NET": "seat_net",
                "TOTAL_BUYRIO": "seat_buy_ratio", "EXPLANATION": "seat_reason",
            })
            seats["code"] = seats["code"].astype(str).str.zfill(6)
            for col in ("seat_buy", "seat_sell", "seat_net", "seat_buy_ratio"):
                if col in seats:
                    seats[col] = pd.to_numeric(seats[col], errors="coerce")
            seats["is_inst"] = seats["seat_name"].astype(str).str.contains("机构专用")
            seats["is_retail"] = seats["seat_name"].astype(str).str.contains("拉萨|东方财富")
            seats["is_hot"] = ~(seats["is_inst"] | seats["is_retail"])
            grouped = seats.groupby("code").agg(
                lhb_seat_count=("seat_name", "nunique"),
                lhb_hot_seat_count=("is_hot", "sum"),
                lhb_inst_count=("is_inst", "sum"),
                lhb_top_buy=("seat_buy", "max"),
                lhb_top_buy_ratio=("seat_buy_ratio", "max"),
                lhb_seat_net=("seat_net", "sum"),
            ).reset_index()
            daily = daily.merge(grouped, on="code", how="left")
            daily["lhb_net_yi"] = daily["lhb_net"] / 1e8
            daily["lhb_buy_yi"] = daily["lhb_buy"] / 1e8
        else:
            for col in ("lhb_seat_count", "lhb_hot_seat_count", "lhb_inst_count", "lhb_top_buy", "lhb_top_buy_ratio", "lhb_seat_net"):
                daily[col] = np.nan
            daily["lhb_net_yi"] = daily["lhb_net"] / 1e8
            daily["lhb_buy_yi"] = daily["lhb_buy"] / 1e8
        daily = daily.drop_duplicates("code", keep="first")
        keep = ["code", "name", "lhb_net_yi", "lhb_buy_yi", "lhb_net_ratio", "lhb_reason",
                "lhb_buy_ratio", "lhb_hot_seat_count", "lhb_inst_count", "lhb_top_buy_ratio"]
        daily = daily[[c for c in keep if c in daily.columns]]
        bundle = LhbBundle(daily, seats, warning="")
        self.cache.set(key, bundle, {"date": day, "rows": int(len(daily))})
        return bundle

    # ------------------------------------------------------------- 公告
    def _ann_one(self, code: str, asof: date) -> Dict[str, Any]:
        key = f"ann_{asof:%Y%m%d}_{code}"
        cached = self.cache.get(key, max_age_sec=12 * 3600)
        if cached is not None:
            return cached.value
        params = {
            "sr": -1, "page_size": 20, "page_index": 1, "ann_type": "A",
            "client_source": "web", "stock_list": str(code).zfill(6),
        }
        out = {"code": str(code).zfill(6), "ann_checked": True, "ann_red": [], "ann_good": [],
               "ann_titles": [], "ann_latest": ""}
        try:
            data = self._get(ANN_URL, params)
            items = ((data.get("data") or {}).get("list") or [])
            reds, goods, titles = [], [], []
            cutoff = pd.Timestamp(asof) + pd.Timedelta(days=1)
            for item in items[:15]:
                notice = str(item.get("notice_date") or "")[:10]
                if notice and pd.Timestamp(notice) > cutoff:
                    continue
                title = str(item.get("title") or item.get("title_ch") or "").strip()
                if not title:
                    continue
                titles.append(f"{notice} {title}")
                for label, pattern in ANN_RED_PATTERNS:
                    if re.search(pattern, title):
                        reds.append(label)
                for label, pattern in ANN_GOOD_PATTERNS:
                    if re.search(pattern, title):
                        goods.append(label)
                if len(titles) >= 6:
                    break
            out["ann_red"] = sorted(set(reds))
            out["ann_good"] = sorted(set(goods))
            out["ann_titles"] = titles
            out["ann_latest"] = titles[0] if titles else ""
        except Exception as exc:  # noqa: BLE001
            out["ann_checked"] = False
            out["ann_error"] = str(exc)[:100]
        self.cache.set(key, out, {"date": str(asof)})
        return out

    def get_announcements(self, codes: Iterable[str], asof: date, force: bool = False) -> AnnBundle:
        codes = sorted({str(c).zfill(6) for c in codes})
        if not codes:
            return AnnBundle(pd.DataFrame())
        rows = []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(codes))) as pool:
            futures = [pool.submit(self._ann_one, code, asof) for code in codes]
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception:
                    continue
        if not rows:
            return AnnBundle(pd.DataFrame(), warning="公告接口全部失败")
        df = pd.DataFrame(rows)
        df["ann_red_text"] = df["ann_red"].map(lambda x: "；".join(x) if isinstance(x, list) else "")
        df["ann_good_text"] = df["ann_good"].map(lambda x: "；".join(x) if isinstance(x, list) else "")
        return AnnBundle(df)


def merge_limitup(features: pd.DataFrame, bundle: LimitUpBundle) -> pd.DataFrame:
    df = features.copy()
    if bundle.pool.empty:
        return df
    cols = ["code", "board_count", "seal_amount_yi", "seal_ratio", "open_count", "sector_limitup_count", "first_time"]
    add = bundle.pool[[c for c in cols if c in bundle.pool.columns]].set_index("code")
    for col in add.columns:
        df[col] = add[col].reindex(df.index)
    if "board_count" in df:
        prior = df.get("consecutive_limit_up", pd.Series(0.0, index=df.index)).fillna(0)
        df["consecutive_limit_up"] = np.maximum(prior, df["board_count"].fillna(0))
    return df


def merge_lhb(features: pd.DataFrame, bundle: LhbBundle) -> pd.DataFrame:
    df = features.copy()
    if bundle.daily.empty:
        return df
    add = bundle.daily.set_index("code")
    for col in add.columns:
        if col == "name":
            continue
        df[col] = add[col].reindex(df.index)
    df["lhb_on_list"] = df.get("lhb_net_yi", pd.Series(np.nan, index=df.index)).notna().astype(float)
    return df


def merge_announcements(df: pd.DataFrame, bundle: AnnBundle) -> pd.DataFrame:
    out = df.copy()
    if bundle.flags.empty:
        out["ann_checked"] = False
        out["ann_red_text"] = ""
        out["ann_good_text"] = ""
        return out
    add = bundle.flags.set_index("code")
    for col in ("ann_checked", "ann_red_text", "ann_good_text", "ann_latest"):
        if col in add:
            out[col] = add[col].reindex(out["code"].astype(str)).values
    if "ann_checked" in out:
        out["ann_checked"] = out["ann_checked"].fillna(False)
    if "ann_red_text" in out:
        out["ann_red_text"] = out["ann_red_text"].fillna("")
    if "ann_good_text" in out:
        out["ann_good_text"] = out["ann_good_text"].fillna("")
    return out


ANN_PENALTY = {
    "澄清/无关联": 18,
    "立案/调查": 25,
    "退市风险": 25,
    "减持计划": 10,
    "问询/监管": 7,
    "业绩暴雷": 9,
}
ANN_HARD = {"澄清/无关联", "立案/调查", "退市风险"}


def apply_announcements(df: pd.DataFrame, bundle: AnnBundle) -> Tuple[pd.DataFrame, List[str]]:
    """把公告红线/催化合并进已排序榜单，并重新排名。

    只对榜单前若干名做公告核查，因此未核查标的会保留 ``ann_checked=False``，
    不能把“未查出”当成“没有风险”。
    """
    if df.empty:
        return df, []
    out = merge_announcements(df, bundle)
    penalties, hard_codes, flags = [], [], []
    for _, row in out.iterrows():
        text = str(row.get("ann_red_text") or "")
        labels = [x for x in re.split(r"[；,，]", text) if x]
        pen = max([ANN_PENALTY.get(x, 0) for x in labels] + [0])
        hard = any(x in ANN_HARD for x in labels)
        penalties.append(float(pen) if not hard else 30.0)
        if hard:
            hard_codes.append(str(row["code"]))
        existing = row.get("risk_flags") if isinstance(row.get("risk_flags"), list) else []
        risk = list(dict.fromkeys(list(existing) + [f"公告：{x}" for x in labels]))
        flags.append(risk)
    out["ann_penalty"] = penalties
    out["ann_hard_excluded"] = out["code"].astype(str).isin(hard_codes)
    out["risk_flags"] = flags
    out["risk_flags_text"] = out["risk_flags"].map(lambda x: "；".join(x) if x else "模型未见硬伤（不代表低风险）")
    out["score"] = (out["score"] - out["ann_penalty"]).clip(lower=0)
    out["confidence"] = (out["confidence"] - out["ann_penalty"] * 0.45).clip(lower=0, upper=100)
    out = out.sort_values(["score", "confidence", "amount20_yi"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, hard_codes


def market_sentiment(limitup: LimitUpBundle, snapshot: pd.DataFrame) -> Dict[str, Any]:
    """市场温度：涨停家数/最高板/涨家占比。只作为环境描述和风险闸门。"""
    out: Dict[str, Any] = {"limit_up_count": 0, "max_board": 0, "multi_board_count": 0,
                           "up_ratio": np.nan, "label": "未知", "source": "snapshot"}
    if snapshot is not None and not snapshot.empty:
        base = snapshot[(~snapshot["is_st"]) & (~snapshot["is_suspended"]) & (snapshot["board"] != "北交所")]
        if len(base):
            out["up_ratio"] = float((base["day_pct_snap"] > 0).mean())
    if not limitup.pool.empty:
        out["limit_up_count"] = int(len(limitup.pool))
        out["max_board"] = int(limitup.pool["board_count"].max())
        out["multi_board_count"] = int((limitup.pool["board_count"] >= 2).sum())
        out["source"] = "limitup"
    up = out["up_ratio"]
    if out["limit_up_count"] >= 80 and (not np.isfinite(up) or up >= 0.55):
        out["label"] = "强"
    elif out["limit_up_count"] >= 45 and (not np.isfinite(up) or up >= 0.45):
        out["label"] = "中等"
    else:
        out["label"] = "偏弱"
    return out
