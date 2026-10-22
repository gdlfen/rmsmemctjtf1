# -*- coding: utf-8 -*-
"""项目级配置、路径与扫描参数。"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DEFAULT_MODEL_PATH = CONFIG_DIR / "model.json"
DEFAULT_MACRO_PATH = CONFIG_DIR / "macro_playbook.json"


def _pick_data_dir() -> Path:
    """优先使用项目 data/，不可写时退到系统临时目录（云端只读文件系统兜底）。"""
    candidates = []
    env = os.environ.get("SPEC_RADAR_DATA_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "data")
    candidates.append(Path(tempfile.gettempdir()) / "ashare_spec_radar")
    for path in candidates:
        try:
            (path / "cache").mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / "ashare_spec_radar"


DATA_DIR = _pick_data_dir()
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScanSettings:
    """一次扫描的全部用户可调参数。"""

    n_potential: int = 12
    n_active: int = 12
    scan_potential: int = 300
    scan_active: int = 300
    boards: Tuple[str, ...] = ("主板", "创业板", "科创板")
    min_float_mcap_yi: float = 0.0
    max_float_mcap_yi: float = 0.0
    min_amount20_yi: float = 0.0
    exclude_loss: bool = False
    risk_mode: str = "auto"  # auto / offensive / defensive
    sort_mode: str = "speculation"  # speculation / rally / fundamental
    as_of: str = ""  # YYYY-MM-DD；留空表示自动取最近已收盘交易日
    lookback_bars: int = 320

    def __post_init__(self) -> None:
        if not 1 <= int(self.n_potential) <= 50:
            raise ValueError("潜在榜 N 必须在 1..50")
        if not 1 <= int(self.n_active) <= 50:
            raise ValueError("活跃榜 N 必须在 1..50")
        if not 100 <= int(self.scan_potential) <= 1200:
            raise ValueError("潜在榜扫描深度必须在 100..1200")
        if not 100 <= int(self.scan_active) <= 1200:
            raise ValueError("活跃榜扫描深度必须在 100..1200")
        if self.risk_mode not in {"auto", "offensive", "defensive"}:
            raise ValueError("risk_mode 非法")
        if self.sort_mode not in {"rally", "rally_max", "fundamental", "speculation", "profitability"}:
            raise ValueError("sort_mode 非法")

    def key(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["boards"] = list(self.boards)
        return data


def load_model_config(path: Path | None = None) -> Dict[str, Any]:
    path = path or DEFAULT_MODEL_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("potential", "active"):
        if key not in cfg:
            raise ValueError(f"模型配置缺少 {key}")
        weights = cfg[key].get("weights", {})
        total = sum(float(v) for v in weights.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"{key} 权重之和应为 1，当前为 {total:.4f}")
    return cfg


def load_macro_playbook(path: Path | None = None) -> Dict[str, Any]:
    path = path or DEFAULT_MACRO_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    themes = cfg.get("themes")
    if not isinstance(themes, list) or not themes:
        raise ValueError("宏观配置缺少 themes")
    ids = [str(t.get("id", "")) for t in themes]
    if len(ids) != len(set(ids)):
        raise ValueError("宏观配置 theme id 重复")
    return cfg


@dataclass
class ScanMeta:
    signal_date: str = ""
    snapshot_time: str = ""
    generated_at: str = ""
    data_source: str = "东方财富公开行情接口 / 腾讯行情（备用）"
    warnings: list[str] = field(default_factory=list)
    universe_total: int = 0
    eligible_base: int = 0
    scanned_potential: int = 0
    scanned_active: int = 0
    elapsed_sec: float = 0.0
    stale: bool = False
    model_version: str = ""
    config_fingerprint: str = ""
