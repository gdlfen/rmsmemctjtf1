# -*- coding: utf-8 -*-
"""离线灾备数据包。

Git 仓库里可携带一份最近一次成功扫描的快照/K 线（data/bootstrap）。
当云端网络暂时无法访问行情接口时，程序仍能打开并展示“最后一次成功数据”，
但 UI 会强制标注数据基准日，避免把旧数据误当实时信号。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = ROOT / "data" / "bootstrap"
META_PATH = BOOTSTRAP_DIR / "meta.json"
SNAPSHOT_PATH = BOOTSTRAP_DIR / "snapshot.parquet"
INDUSTRY_PATH = BOOTSTRAP_DIR / "boards_industry.parquet"
CONCEPT_PATH = BOOTSTRAP_DIR / "boards_concept.parquet"
STOCK_KLINE_PATH = BOOTSTRAP_DIR / "stock_kline.parquet"
MARKET_KLINE_PATH = BOOTSTRAP_DIR / "market_kline.parquet"
BOARD_KLINE_PATH = BOOTSTRAP_DIR / "board_kline.parquet"


def available() -> bool:
    return META_PATH.exists() and SNAPSHOT_PATH.exists()


def load_meta() -> Dict[str, Any]:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _restore_concepts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "concepts" in d and d["concepts"].dtype == object:
        d["concepts"] = d["concepts"].map(
            lambda x: [str(v).strip() for v in str(x).split("|") if str(v).strip()] if not isinstance(x, list) else x
        )
    return d


def load_snapshot() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError("缺少 data/bootstrap/snapshot.parquet")
    df = pd.read_parquet(SNAPSHOT_PATH)
    df["code"] = df["code"].astype(str).str.zfill(6)
    return _restore_concepts(df)


def load_boards(kind: str = "industry") -> pd.DataFrame:
    path = INDUSTRY_PATH if kind == "industry" else CONCEPT_PATH
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_stock_panel(
    codes: Optional[list[str]] = None,
    end_date: Optional[str] = None,
    lmt: int = 320,
) -> pd.DataFrame:
    if not STOCK_KLINE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(STOCK_KLINE_PATH)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    if codes:
        df = df[df["code"].isin({str(c).zfill(6) for c in codes})]
    if end_date:
        end = pd.Timestamp(end_date)
        df = df[df["date"] <= end]
    if df.empty:
        return df
    df = df.sort_values(["code", "date"]).groupby("code", group_keys=False).tail(int(lmt)).reset_index(drop=True)
    return df


def load_board_panel(
    secids: Optional[list[str]] = None,
    end_date: Optional[str] = None,
    lmt: int = 130,
) -> pd.DataFrame:
    if not BOARD_KLINE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(BOARD_KLINE_PATH)
    df["secid"] = df["secid"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    if secids:
        df = df[df["secid"].isin({str(s) for s in secids})]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    if df.empty:
        return df
    return df.sort_values(["secid", "date"]).groupby("secid", group_keys=False).tail(int(lmt)).reset_index(drop=True)


def load_market_panel(secid: Optional[str] = None, end_date: Optional[str] = None, lmt: int = 320) -> pd.DataFrame:
    if not MARKET_KLINE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(MARKET_KLINE_PATH)
    df["secid"] = df["secid"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    if secid:
        df = df[df["secid"] == secid]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    if df.empty:
        return df
    df = df.sort_values(["secid", "date"]).groupby("secid", group_keys=False).tail(int(lmt)).reset_index(drop=True)
    return df
