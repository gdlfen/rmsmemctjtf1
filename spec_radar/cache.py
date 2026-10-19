# -*- coding: utf-8 -*-
"""轻量磁盘缓存：raw 数据与完整扫描结果都能在接口短暂失败时兜底。"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import CACHE_DIR


def safe_key(key: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(key))[:120]
    return text or hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheEntry:
    value: Any
    saved_at: float
    meta: dict

    @property
    def age_sec(self) -> float:
        return max(0.0, time.time() - self.saved_at)


class DiskCache:
    def __init__(self, directory: Path = CACHE_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.directory / f"{safe_key(key)}.pkl"

    def get(self, key: str, max_age_sec: Optional[float] = None) -> Optional[CacheEntry]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            entry = CacheEntry(value=payload["value"], saved_at=float(payload["saved_at"]), meta=payload.get("meta", {}))
        except Exception:
            return None
        if max_age_sec is not None and entry.age_sec > float(max_age_sec):
            return None
        return entry

    def set(self, key: str, value: Any, meta: Optional[dict] = None) -> None:
        path = self._path(key)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        payload = {"saved_at": time.time(), "value": value, "meta": meta or {}}
        with self._lock:
            try:
                with tmp.open("wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError:
            pass

    def clear(self) -> int:
        count = 0
        for path in self.directory.glob("*.pkl"):
            try:
                path.unlink()
                count += 1
            except OSError:
                continue
        return count
