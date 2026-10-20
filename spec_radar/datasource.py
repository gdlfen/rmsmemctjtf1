# -*- coding: utf-8 -*-
"""A 股行情数据访问层。

主源：东方财富公开行情接口（全市场快照、行业/概念板块、日 K）。
K 线备用：腾讯前复权日 K（仅当东财单只失败时使用，成交额用均价*成交量估算）。

设计原则：
1. 所有请求带超时、重试和主机轮换；
2. 全市场快照分页抓取并校验完整性，避免“抓了一半就排名”；
3. 成功结果落盘缓存；接口临时失败时允许使用旧缓存，并在 UI 明确标注；
4. 不在数据层做任何选股逻辑，保证可测试、可替换。
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from .cache import CacheEntry, DiskCache
from .config import CACHE_DIR

LOG = logging.getLogger(__name__)

CN_TZ = timezone(timedelta(hours=8))
STOCK_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
SNAPSHOT_FIELDS = (
    "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,"
    "f22,f23,f24,f25,f26,f37,f38,f39,f40,f41,f45,f46,f48,f49,f62,f100,"
    "f102,f103,f115,f124"
)
BOARD_FIELDS = "f2,f3,f8,f12,f14,f20,f24,f25,f62,f104,f105,f128,f136,f140"
INDEXES: Tuple[Tuple[str, str], ...] = (
    ("1.000001", "上证指数"),
    ("1.000300", "沪深300"),
    ("0.399001", "深证成指"),
    ("0.399006", "创业板指"),
    ("1.000688", "科创50"),
    ("1.000852", "中证1000"),
)
PROXIES: Tuple[Tuple[str, str], ...] = (
    ("1.518880", "黄金ETF"),
    ("1.513100", "纳指ETF"),
    ("1.513130", "恒生科技ETF"),
    ("1.511010", "国债ETF"),
    ("1.512400", "有色ETF"),
    ("1.512480", "半导体ETF"),
    ("1.512660", "军工ETF"),
    ("0.159992", "创新药ETF"),
    ("1.512000", "券商ETF"),
    ("1.562500", "机器人ETF"),
    ("1.515790", "光伏ETF"),
    ("1.515220", "煤炭ETF"),
    ("1.510880", "红利ETF"),
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DataError(RuntimeError):
    """行情接口不可用且没有可接受的缓存。"""


def china_now() -> datetime:
    return datetime.now(CN_TZ)


def bootstrap_datetime() -> datetime:
    """内置离线包的基准时间，避免把旧快照显示成“当前快照”。"""
    try:
        from . import bootstrap
        meta = bootstrap.load_meta()
        for key, fmt in (("snapshot_time", "%Y-%m-%d %H:%M"), ("built_at", "%Y-%m-%d %H:%M:%S")):
            value = str(meta.get(key, "")).strip()
            if value:
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=CN_TZ)
                except ValueError:
                    continue
    except Exception:
        pass
    return china_now()


def fmt_date(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "")[:8]
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    return value.strftime("%Y%m%d")


def market_of_code(code: str) -> int:
    code = str(code).zfill(6)
    if code.startswith(("5", "6", "9")):
        return 1
    return 0


def secid_of_code(code: str) -> str:
    code = str(code).zfill(6)
    return f"{market_of_code(code)}.{code}"


def tencent_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return ("sh" if market_of_code(code) == 1 else "sz") + code


def board_of_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("4", "8", "92")):
        return "北交所"
    return "主板"


def limit_pct_of_code(code: str, name: str = "") -> float:
    name = str(name)
    if "ST" in name.upper():
        return 0.05
    board = board_of_code(code)
    if board in ("创业板", "科创板"):
        return 0.20
    if board == "北交所":
        return 0.30
    return 0.10


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("-", np.nan), errors="coerce")


def normalize_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """把东财字段名转换为稳定的内部列名，并补全板块、交易日历等派生字段。"""
    if raw.empty:
        raise DataError("全市场快照为空")
    rename = {
        "f2": "price", "f3": "day_pct", "f5": "volume", "f6": "amount",
        "f8": "turnover", "f9": "pe_dyn", "f10": "volume_ratio",
        "f12": "code", "f13": "market", "f14": "name", "f15": "high",
        "f16": "low", "f17": "open", "f18": "prev_close",
        "f20": "total_mcap", "f21": "float_mcap", "f23": "pb",
        "f24": "ret60_snap", "f25": "ret_ytd_snap", "f26": "list_date",
        "f37": "roe", "f38": "total_shares", "f39": "float_shares",
        "f40": "revenue", "f41": "rev_yoy", "f45": "net_profit",
        "f46": "profit_yoy", "f48": "eps", "f49": "gross_margin",
        "f62": "main_inflow", "f100": "industry", "f102": "region",
        "f103": "concepts_raw", "f115": "pe_ttm", "f124": "timestamp",
    }
    df = raw.rename(columns=rename).copy()
    for col in df.columns:
        if col not in {"code", "name", "industry", "region", "concepts_raw", "list_date"}:
            df[col] = _num(df[col])

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str).str.replace(r"\s+", "", regex=True)
    df["industry"] = df["industry"].fillna("未分类").astype(str).replace("-", "未分类")
    df["region"] = df["region"].fillna("").astype(str)
    list_nums = pd.to_numeric(df["list_date"].replace("-", np.nan), errors="coerce")
    df["list_date"] = pd.to_datetime(
        list_nums.round().astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    df["concepts"] = df["concepts_raw"].fillna("").astype(str).map(
        lambda s: [x.strip() for x in s.split(",") if x.strip() and x.strip() != "-"]
    )
    df["is_st"] = df["name"].str.upper().str.contains("ST", regex=False, na=False) | df["name"].str.contains("退", regex=False, na=False)
    df["board"] = df["code"].map(board_of_code)
    df["limit_pct"] = [limit_pct_of_code(c, n) for c, n in zip(df["code"], df["name"])]
    df["is_20pct"] = df["limit_pct"] >= 0.195
    df["secid"] = df["code"].map(secid_of_code)
    df["is_suspended"] = df["price"].isna() | df["amount"].isna() | (df["amount"] <= 0)
    df["turnover_snap"] = df["turnover"] / 100.0
    df["day_pct_snap"] = df["day_pct"] / 100.0
    df["ret60_snap"] = df["ret60_snap"] / 100.0
    df["ret_ytd_snap"] = df["ret_ytd_snap"] / 100.0
    df["main_inflow_yi"] = df["main_inflow"] / 1e8
    df["float_mcap_yi"] = df["float_mcap"] / 1e8
    df["total_mcap_yi"] = df["total_mcap"] / 1e8
    # 52 周高低、财务等不是快照主字段，后续由 K 线或财务报表补。
    if "float_shares" not in df or df["float_shares"].isna().all():
        df["float_shares"] = df["float_mcap"] / df["price"]
    df = df[df["code"].str.match(r"^\d{6}$", na=False)].copy()
    df = df.drop_duplicates("code", keep="first").reset_index(drop=True)
    return df


def normalize_boards(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["board_code", "name", "pct", "ret60", "ret_ytd", "turnover", "total_mcap", "main_inflow", "up_count", "down_count", "leader"])
    rename = {
        "f2": "close", "f3": "pct", "f8": "turnover", "f12": "board_code",
        "f14": "name", "f20": "total_mcap", "f24": "ret60", "f25": "ret_ytd",
        "f62": "main_inflow", "f104": "up_count", "f105": "down_count",
        "f128": "leader", "f136": "leader_pct", "f140": "leader_code",
    }
    df = raw.rename(columns=rename).copy()
    for col in ("close", "pct", "turnover", "total_mcap", "ret60", "ret_ytd", "main_inflow", "up_count", "down_count", "leader_pct"):
        if col in df:
            df[col] = _num(df[col])
    df["name"] = df["name"].astype(str).str.replace(r"\s+", "", regex=True)
    df["kind"] = kind
    df["pct"] = df["pct"] / 100.0
    df["ret60"] = df["ret60"] / 100.0
    df["ret_ytd"] = df["ret_ytd"] / 100.0
    df["turnover"] = df["turnover"] / 100.0
    df = df.sort_values("total_mcap", ascending=False, na_position="last")
    df = df.drop_duplicates("name", keep="first").reset_index(drop=True)
    return df


@dataclass
class SnapshotBundle:
    df: pd.DataFrame
    fetched_at: datetime
    stale: bool
    warnings: List[str] = field(default_factory=list)
    source: str = "eastmoney"


@dataclass
class BoardBundle:
    industry: pd.DataFrame
    concept: pd.DataFrame
    fetched_at: datetime
    stale: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class KlineBundle:
    frame: pd.DataFrame
    fetched_at: datetime
    failed: List[str] = field(default_factory=list)
    fallback: List[str] = field(default_factory=list)
    stale: bool = False
    warnings: List[str] = field(default_factory=list)


class MarketClient:
    def __init__(self, cache: DiskCache | None = None, workers: int = 10, timeout: float = 18.0) -> None:
        self.cache = cache or DiskCache(CACHE_DIR)
        self.workers = max(2, int(workers))
        self.timeout = float(timeout)
        self._local = threading.local()
        self._host_lock = threading.Lock()
        self._preferred_host = "push2.eastmoney.com"
        self._kline_fail_streak = 0
        self._kline_down_until = 0.0
        self._tencent_down_until = 0.0
        self._eastmoney_kline_down_until = 0.0
        self.offline = str(os.environ.get("SPEC_RADAR_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.last_errors: List[str] = []

    # ------------------------------------------------------------------ HTTP
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
            session.mount("https://", adapter)
            session.headers.update({
                "User-Agent": UA,
                "Referer": "https://quote.eastmoney.com/",
                "Accept": "application/json,text/plain,*/*",
            })
            self._local.session = session
        return session

    def _get_json(
        self, host: str, path: str, params: Dict[str, Any], attempts: int = 3,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        url = f"https://{host}{path}"
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._session().get(url, params=params, timeout=timeout or self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("rc") not in (0, None):
                    raise RuntimeError(f"rc={payload.get('rc')}")
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(1.8, 0.25 * (2 ** attempt)))
        raise DataError(f"请求 {host}{path} 失败: {type(last_error).__name__}: {str(last_error)[:120]}")

    def _paged_clist(self, fs: str, fields: str, page_size: int = 100, max_pages: int = 80) -> pd.DataFrame:
        """分页抓取 clist。pz 最大 100，按代码升序避免行情跳动造成跨页重复。"""
        base_params = {
            "pz": page_size, "po": 0, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f12", "fs": fs, "fields": fields,
        }
        first = None
        first_err = None
        for host in ("push2.eastmoney.com", "82.push2.eastmoney.com"):
            try:
                payload = self._get_json(
                    host, "/api/qt/clist/get", dict(base_params, pn=1),
                    attempts=2, timeout=10.0,
                )
                data = payload.get("data") or {}
                diff = data.get("diff") or []
                if diff:
                    first = (host, data)
                    with self._host_lock:
                        self._preferred_host = host
                    break
            except Exception as exc:  # noqa: BLE001
                first_err = exc
        if first is None:
            raise DataError(f"全市场列表请求失败: {first_err}")
        host, data = first
        total = int(data.get("total") or 0)
        rows: List[Dict[str, Any]] = list(data.get("diff") or [])
        page_count = min(max_pages, int(math.ceil(total / page_size))) if total else 1
        if page_count > 1:
            def fetch_page(pn: int) -> List[Dict[str, Any]]:
                payload = self._get_json(
                    host, "/api/qt/clist/get", dict(base_params, pn=pn),
                    attempts=2, timeout=12.0,
                )
                return list((payload.get("data") or {}).get("diff") or [])

            with ThreadPoolExecutor(max_workers=min(10, self.workers)) as pool:
                futures = {pool.submit(fetch_page, pn): pn for pn in range(2, page_count + 1)}
                for fut in as_completed(futures):
                    pn = futures[fut]
                    try:
                        part = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        raise DataError(f"全市场列表第 {pn} 页失败: {exc}") from exc
                    if not part:
                        raise DataError(f"全市场列表第 {pn} 页为空（可能被限流）")
                    rows.extend(part)
        df = pd.DataFrame(rows)
        if df.empty:
            raise DataError("全市场列表返回空")
        df = df.drop_duplicates(subset=["f12"], keep="first")
        if total and len(df) < total * 0.97:
            raise DataError(f"全市场快照不完整：拿到 {len(df)} / {total}")
        return df

    # -------------------------------------------------------------- snapshot
    def get_snapshot(self, force: bool = False, ttl_sec: float = 1200) -> SnapshotBundle:
        key = "snapshot_all_a"
        cached = None if force else self.cache.get(key, max_age_sec=ttl_sec)
        if self.offline:
            offline_cache = self.cache.get(key, max_age_sec=None)
            if offline_cache is not None and isinstance(offline_cache.value, pd.DataFrame) and not offline_cache.value.empty:
                return SnapshotBundle(
                    df=offline_cache.value.copy(),
                    fetched_at=datetime.fromtimestamp(offline_cache.saved_at, CN_TZ),
                    stale=True,
                    warnings=["离线模式：使用本机快照缓存，不访问实时接口。"],
                )
            from . import bootstrap
            if bootstrap.available():
                return SnapshotBundle(
                    df=bootstrap.load_snapshot(), fetched_at=bootstrap_datetime(), stale=True,
                    warnings=["离线模式：使用内置离线快照。"],
                )
            raise DataError("离线模式已开启，但本机没有快照缓存或内置离线数据")
        if cached is not None and isinstance(cached.value, pd.DataFrame) and not cached.value.empty:
            return SnapshotBundle(
                df=cached.value.copy(),
                fetched_at=datetime.fromtimestamp(cached.saved_at, CN_TZ),
                stale=False,
                source="eastmoney/cache",
            )
        try:
            raw = self._paged_clist(STOCK_FS, SNAPSHOT_FIELDS)
            df = normalize_snapshot(raw)
            self.cache.set(key, df, {"source": "eastmoney", "rows": int(len(df))})
            return SnapshotBundle(df=df, fetched_at=china_now(), stale=False)
        except Exception as exc:  # noqa: BLE001
            stale = self.cache.get(key, max_age_sec=None)
            if stale is not None and isinstance(stale.value, pd.DataFrame) and not stale.value.empty:
                age_min = stale.age_sec / 60
                return SnapshotBundle(
                    df=stale.value.copy(),
                    fetched_at=datetime.fromtimestamp(stale.saved_at, CN_TZ),
                    stale=True,
                    warnings=[f"全市场快照接口失败，使用 {age_min:.0f} 分钟前的缓存：{exc}"],
                )
            try:
                from . import bootstrap
                if bootstrap.available():
                    meta = bootstrap.load_meta()
                    return SnapshotBundle(
                        df=bootstrap.load_snapshot(),
                        fetched_at=bootstrap_datetime(),
                        stale=True,
                        warnings=[
                            f"实时接口不可用，使用内置离线快照（构建于 {meta.get('built_at','未知')}，"
                            f"信号基准 {meta.get('signal_date','未知')}）。这不是实时行情。"
                        ],
                    )
            except Exception:
                pass
            raise DataError(f"无法获取 A 股全市场快照，且无可用缓存：{exc}") from exc

    # ---------------------------------------------------------------- boards
    def get_boards(self, force: bool = False, ttl_sec: float = 6 * 3600) -> BoardBundle:
        key_ind = "boards_industry"
        key_con = "boards_concept"
        if self.offline:
            from . import bootstrap
            ind = bootstrap.load_boards("industry")
            con = bootstrap.load_boards("concept")
            if not ind.empty and not con.empty:
                return BoardBundle(industry=ind, concept=con, fetched_at=china_now(), stale=True,
                                   warnings=["离线模式：使用内置板块映射。"])
            cached_ind = self.cache.get(key_ind, max_age_sec=None)
            cached_con = self.cache.get(key_con, max_age_sec=None)
            if cached_ind and cached_con:
                return BoardBundle(industry=cached_ind.value.copy(), concept=cached_con.value.copy(),
                                   fetched_at=china_now(), stale=True,
                                   warnings=["离线模式：使用本机板块缓存。"])
            return BoardBundle(industry=pd.DataFrame(), concept=pd.DataFrame(), fetched_at=china_now(), stale=True,
                               warnings=["离线模式且无板块数据，主题热度按中性处理。"])
        cached_ind = None if force else self.cache.get(key_ind, max_age_sec=ttl_sec)
        cached_con = None if force else self.cache.get(key_con, max_age_sec=ttl_sec)
        if cached_ind and cached_con and isinstance(cached_ind.value, pd.DataFrame) and isinstance(cached_con.value, pd.DataFrame):
            return BoardBundle(
                industry=cached_ind.value.copy(),
                concept=cached_con.value.copy(),
                fetched_at=datetime.fromtimestamp(min(cached_ind.saved_at, cached_con.saved_at), CN_TZ),
                stale=False,
            )
        try:
            raw_ind = self._paged_clist("m:90+t:2", BOARD_FIELDS, max_pages=12)
            raw_con = self._paged_clist("m:90+t:3", BOARD_FIELDS, max_pages=12)
            ind = normalize_boards(raw_ind, "industry")
            con = normalize_boards(raw_con, "concept")
            self.cache.set(key_ind, ind, {"source": "eastmoney"})
            self.cache.set(key_con, con, {"source": "eastmoney"})
            return BoardBundle(industry=ind, concept=con, fetched_at=china_now())
        except Exception as exc:  # noqa: BLE001
            stale_ind = self.cache.get(key_ind, max_age_sec=None)
            stale_con = self.cache.get(key_con, max_age_sec=None)
            if stale_ind and stale_con:
                return BoardBundle(
                    industry=stale_ind.value.copy(), concept=stale_con.value.copy(),
                    fetched_at=datetime.fromtimestamp(min(stale_ind.saved_at, stale_con.saved_at), CN_TZ),
                    stale=True,
                    warnings=[f"板块接口失败，使用旧缓存：{exc}"],
                )
            try:
                from . import bootstrap
                if bootstrap.available():
                    ind = bootstrap.load_boards("industry")
                    con = bootstrap.load_boards("concept")
                    if not ind.empty and not con.empty:
                        return BoardBundle(
                            industry=ind, concept=con, fetched_at=china_now(), stale=True,
                            warnings=["板块接口不可用，使用内置离线板块映射；主题热度按旧数据计算。"],
                        )
            except Exception:
                pass
            empty = pd.DataFrame(columns=["board_code", "name", "pct", "ret60", "ret_ytd", "turnover", "total_mcap", "main_inflow", "up_count", "down_count", "leader"])
            return BoardBundle(
                industry=empty.copy(), concept=empty.copy(), fetched_at=china_now(),
                stale=True, warnings=[f"板块接口不可用，主题热度降级为中性：{exc}"],
            )

    # ----------------------------------------------------------------- kline
    def fetch_kline(
        self,
        secid: str,
        end_date: date | datetime | str,
        lmt: int = 320,
        symbol: str | None = None,
        allow_tencent: bool = True,
    ) -> Tuple[pd.DataFrame, str]:
        params = {
            "secid": secid, "klt": 101, "fqt": 1, "lmt": int(lmt),
            "end": fmt_date(end_date),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        last_error: Exception | None = None
        # 多源顺序：东财（字段最全）→ 腾讯（前复权）→ 新浪（无复权，最后兜底）。
        # 每个源有熔断，避免 840 只股票 × 多次超时把首屏拖到几分钟。
        if time.time() >= self._eastmoney_kline_down_until:
            for attempt in range(2):
                try:
                    payload = self._get_json(
                        "push2his.eastmoney.com", "/api/qt/stock/kline/get", params,
                        attempts=1, timeout=10.0,
                    )
                    data = payload.get("data") or {}
                    klines = data.get("klines") or []
                    if klines:
                        with self._host_lock:
                            self._kline_fail_streak = max(0, self._kline_fail_streak - 1)
                        return parse_eastmoney_klines(klines), "eastmoney"
                    last_error = DataError("K 线为空")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                time.sleep(0.2 * (attempt + 1))
            with self._host_lock:
                self._kline_fail_streak += 1
                if self._kline_fail_streak >= 3:
                    self._eastmoney_kline_down_until = time.time() + 300
        if allow_tencent and symbol and time.time() >= self._tencent_down_until:
            try:
                return self._fetch_kline_tencent(symbol, end_date, lmt), "tencent"
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                with self._host_lock:
                    self._tencent_down_until = time.time() + 600
        if symbol:
            try:
                return self._fetch_kline_sina(symbol, end_date, lmt), "sina"
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise DataError(f"{secid} K 线获取失败: {last_error}")

    def _fetch_kline_tencent(self, symbol: str, end_date: date | datetime | str, lmt: int) -> pd.DataFrame:
        params = {"param": f"{symbol},day,,,{int(lmt)},qfq"}
        payload = self._get_json(
            "web.ifzq.gtimg.cn", "/appstock/app/fqkline/get", params,
            attempts=2,
        )
        node = (payload.get("data") or {}).get(symbol) or {}
        rows = node.get("qfqday") or node.get("day") or []
        if not rows:
            raise DataError(f"腾讯 K 线为空: {symbol}")
        parsed = []
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        for row in rows:
            if len(row) < 6 or str(row[0]) > end_s:
                continue
            try:
                dt = pd.Timestamp(str(row[0]))
                op, cl, hi, lo, vol = (float(row[i]) for i in range(1, 6))
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(x) and x > 0 for x in (op, cl, hi, lo, vol)):
                continue
            amount = vol * 100.0 * ((op + cl) / 2.0)
            parsed.append({
                "date": dt, "open": op, "close": cl, "high": hi, "low": lo,
                "volume": vol, "amount": amount, "amplitude": (hi - lo) / max(cl, 1e-9),
                "pct": 0.0, "change": 0.0, "turnover": np.nan,
            })
        df = pd.DataFrame(parsed)
        if df.empty:
            raise DataError(f"腾讯 K 线解析为空: {symbol}")
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df["pct"] = df["close"].pct_change() * 100.0
        df["change"] = df["close"].diff()
        return df

    def _fetch_kline_sina(self, symbol: str, end_date: date | datetime | str, lmt: int) -> pd.DataFrame:
        params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": min(1023, int(lmt))}
        resp = self._session().get(
            "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData",
            params=params, timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            raise DataError(f"新浪 K 线为空: {symbol}")
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        rows = []
        for item in payload:
            day = str(item.get("day", ""))[:10]
            if not day or day > end_s:
                continue
            try:
                op, hi, lo, cl = (float(item[k]) for k in ("open", "high", "low", "close"))
                vol = float(item.get("volume", 0) or 0)
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(x) and x > 0 for x in (op, hi, lo, cl)) or vol <= 0:
                continue
            amount = vol * ((op + cl) / 2.0)  # 股数 × 均价，近似成交额
            rows.append({
                "date": pd.Timestamp(day), "open": op, "close": cl, "high": hi, "low": lo,
                "volume": vol / 100.0, "amount": amount,
                "amplitude": (hi - lo) / max(cl, 1e-9), "pct": 0.0, "change": 0.0,
                "turnover": np.nan,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise DataError(f"新浪 K 线解析为空: {symbol}")
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df["pct"] = df["close"].pct_change() * 100.0
        df["change"] = df["close"].diff()
        return df

    def get_kline_panel(
        self,
        requests_: Dict[str, str],
        end_date: date | datetime | str,
        lmt: int = 320,
        force: bool = False,
        allow_stale: bool = True,
    ) -> KlineBundle:
        """批量抓取日 K。requests_ 为 {secid: 代码或腾讯 symbol}。"""
        secids = sorted({str(k) for k in requests_.keys() if k})
        key_text = "|".join(secids) + f"|{fmt_date(end_date)}|{lmt}"
        key = "kline_" + hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:20]
        if self.offline:
            boot = self._bootstrap_kline(requests_, end_date, lmt)
            if boot is not None:
                return boot
            cached_offline = self.cache.get(key, max_age_sec=None)
            if cached_offline is not None and isinstance(cached_offline.value, pd.DataFrame):
                return KlineBundle(frame=cached_offline.value.copy(), fetched_at=china_now(), stale=True)
            raise DataError("离线模式且没有可用的内置/缓存 K 线")
        cached = None if force else self.cache.get(key, max_age_sec=6 * 3600)
        if cached is not None and isinstance(cached.value, pd.DataFrame) and not cached.value.empty:
            return KlineBundle(frame=cached.value.copy(), fetched_at=datetime.fromtimestamp(cached.saved_at, CN_TZ))

        frames: List[pd.DataFrame] = []
        failed: List[str] = []
        fallback: List[str] = []
        lock = threading.Lock()

        def task(secid: str) -> None:
            code = requests_.get(secid, "")
            symbol = tencent_symbol(code) if code else None
            try:
                frame, source = self.fetch_kline(secid, end_date, lmt=lmt, symbol=symbol)
                frame = frame.copy()
                frame["secid"] = secid
                frame["code"] = str(code).zfill(6) if code else secid
                frame["source"] = source
                with lock:
                    frames.append(frame)
                    if source in ("tencent", "sina"):
                        fallback.append(secid)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failed.append(secid)
                    if len(self.last_errors) < 30:
                        self.last_errors.append(f"{secid}: {exc}")

        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(secids)))) as pool:
            futures = [pool.submit(task, secid) for secid in secids]
            for fut in as_completed(futures):
                fut.result()

        bootstrap_notes: List[str] = []
        if failed:
            # 实时源有缺口时，用内置灾备补单只/单指数，避免一条缺失就报“指数缺失”。
            fill_req = {secid: requests_.get(secid, "") for secid in failed}
            fill = self._bootstrap_kline(fill_req, end_date, lmt)
            if fill is not None and not fill.frame.empty:
                fill_frame = fill.frame.copy()
                fill_frame = fill_frame[fill_frame["secid"].astype(str).isin(set(failed))] if "secid" in fill_frame else fill_frame
                if not fill_frame.empty:
                    frames.append(fill_frame)
                    filled = set(fill_frame["secid"].astype(str))
                    failed = [s for s in failed if s not in filled]
                    bootstrap_notes.append(f"{len(filled)} 个资产用内置离线包补齐")

        if not frames:
            stale = self.cache.get(key, max_age_sec=None)
            if allow_stale and stale is not None and isinstance(stale.value, pd.DataFrame):
                return KlineBundle(
                    frame=stale.value.copy(),
                    fetched_at=datetime.fromtimestamp(stale.saved_at, CN_TZ),
                    stale=True, failed=failed,
                )
            boot = self._bootstrap_kline(requests_, end_date, lmt)
            if boot is not None:
                return boot
            raise DataError(f"全部 K 线抓取失败（{len(secids)} 只），请检查网络或稍后重试")
        if len(failed) > max(5, int(len(secids) * 0.12)):
            stale = self.cache.get(key, max_age_sec=None)
            if allow_stale and stale is not None and isinstance(stale.value, pd.DataFrame):
                return KlineBundle(
                    frame=stale.value.copy(),
                    fetched_at=datetime.fromtimestamp(stale.saved_at, CN_TZ),
                    stale=True, failed=failed, fallback=fallback,
                )
            boot = self._bootstrap_kline(requests_, end_date, lmt)
            if boot is not None:
                return boot
            raise DataError(f"K 线缺失过多：失败 {len(failed)} / {len(secids)}")
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
        panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
        # 有失败时也缓存可用部分，下次刷新会补抓。
        self.cache.set(key, panel, {"failed": len(failed), "rows": int(len(panel)), "fallback": len(fallback)})
        return KlineBundle(
            frame=panel, fetched_at=china_now(), failed=failed, fallback=fallback,
            warnings=bootstrap_notes,
        )

    def _bootstrap_kline(
        self,
        requests_: Dict[str, str],
        end_date: date | datetime | str,
        lmt: int,
    ) -> Optional[KlineBundle]:
        try:
            from . import bootstrap
            if not bootstrap.available():
                return None
            market_ids = {secid for secid, _ in INDEXES + PROXIES}
            if set(requests_.keys()) <= market_ids:
                market = bootstrap.load_market_panel(None, fmt_date(end_date), lmt)
                if not market.empty:
                    market = market[market["secid"].astype(str).isin(set(requests_.keys()))].copy()
                    if not market.empty:
                        return KlineBundle(
                            frame=market.reset_index(drop=True), fetched_at=china_now(),
                            stale=True, warnings=["离线数据：指数/ETF 来自内置灾备包。"],
                        )
                return None
            # 行业/概念板块的 secid 以 90. 开头。
            if all(str(secid).startswith("90.") for secid in requests_.keys()):
                board = bootstrap.load_board_panel(list(requests_.keys()), fmt_date(end_date), lmt)
                if not board.empty:
                    return KlineBundle(
                        frame=board.reset_index(drop=True), fetched_at=china_now(),
                        stale=True, warnings=["离线数据：行业/概念板块指数来自内置灾备包。"],
                    )
                return None
            codes = [str(code).zfill(6) for code in requests_.values() if str(code).strip()]
            if not codes:
                return None
            stock = bootstrap.load_stock_panel(codes, fmt_date(end_date), lmt)
            if not stock.empty:
                reverse = {str(code).zfill(6): secid for secid, code in requests_.items()}
                if "secid" not in stock.columns:
                    stock["secid"] = stock["code"].map(reverse)
                if "source" not in stock.columns:
                    stock["source"] = "bootstrap"
                meta = bootstrap.load_meta()
                available_codes = set(stock["code"].astype(str).str.zfill(6))
                missing = [secid for secid, code in requests_.items() if str(code).zfill(6) not in available_codes]
                return KlineBundle(
                    frame=stock.sort_values(["code", "date"]).reset_index(drop=True),
                    fetched_at=china_now(),
                    failed=missing,
                    stale=True,
                    warnings=[
                        f"实时 K 线不可用，使用内置离线数据（构建于 {meta.get('built_at','未知')}，"
                        f"信号基准 {meta.get('signal_date','未知')}）。这不是实时行情。"
                    ],
                )
            market = bootstrap.load_market_panel(None, fmt_date(end_date), lmt)
            if not market.empty:
                market = market[market["secid"].astype(str).isin(set(requests_.keys()))].copy()
                if not market.empty:
                    return KlineBundle(
                        frame=market.reset_index(drop=True), fetched_at=china_now(),
                        stale=True,
                        warnings=["实时指数/ETF K 线不可用，使用内置离线数据。"],
                    )
        except Exception:
            return None
        return None

    def get_market_probe(self, end_date: date | datetime | str | None = None, lmt: int = 45) -> Tuple[pd.DataFrame, datetime]:
        end = fmt_date(end_date) if end_date else fmt_date(china_now().date())
        if self.offline:
            from . import bootstrap
            frame = bootstrap.load_market_panel("1.000001", end, lmt)
            if not frame.empty:
                return frame, china_now()
            raise DataError("离线模式：内置数据缺少上证指数")
        try:
            frame, _ = self.fetch_kline("1.000001", end, lmt=lmt, symbol="sh000001", allow_tencent=True)
            return frame, china_now()
        except Exception as exc:  # noqa: BLE001
            try:
                from . import bootstrap
                frame = bootstrap.load_market_panel("1.000001", end, lmt)
                if not frame.empty:
                    return frame, china_now()
            except Exception:
                pass
            raise DataError(f"无法获取上证指数基准：{exc}") from exc

    def get_market_context_panel(self, end_date: date | datetime | str, lmt: int = 320) -> KlineBundle:
        mapping: Dict[str, str] = {}
        for secid, _name in INDEXES + PROXIES:
            mapping[secid] = secid.split(".")[1]
        return self.get_kline_panel(mapping, end_date, lmt=lmt)


def parse_eastmoney_klines(klines: Sequence[str]) -> pd.DataFrame:
    rows = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 11:
            continue
        try:
            rows.append({
                "date": pd.Timestamp(parts[0]),
                "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]),
                "volume": float(parts[5]), "amount": float(parts[6]),
                "amplitude": float(parts[7]), "pct": float(parts[8]),
                "change": float(parts[9]), "turnover": float(parts[10]),
            })
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "open", "close", "high", "low"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def determine_signal_date(client: MarketClient, requested: str = "", probe: pd.DataFrame | None = None) -> Tuple[date, pd.DataFrame]:
    """以最近一根已收盘日 K 作为信号基准日。

    盘中（15:10 前）即使接口返回了当天 K 线，也丢弃，避免用未完成 K 线排名。
    """
    now = china_now()
    if requested:
        try:
            req = datetime.strptime(requested, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("as_of 必须是 YYYY-MM-DD") from exc
    else:
        req = now.date()
    if probe is None:
        frame, _ = client.get_market_probe(req, lmt=45)
    else:
        frame = probe
    if frame is None or frame.empty:
        raise DataError("无法确定最近交易日（上证指数 K 线为空）")
    frame = frame[frame["date"].dt.date <= req].copy()
    if frame.empty:
        raise DataError(f"{req} 之前没有可用交易日")
    latest = pd.Timestamp(frame["date"].iloc[-1]).date()
    # 只有请求“今天”且当前仍在盘中时才需要丢弃尚未收盘的当日 K。
    if not requested and latest == now.date() and now.hour * 60 + now.minute < 15 * 60 + 10:
        if len(frame) >= 2:
            latest = pd.Timestamp(frame["date"].iloc[-2]).date()
            frame = frame.iloc[:-1].copy()
    return latest, frame
