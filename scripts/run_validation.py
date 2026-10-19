# -*- coding: utf-8 -*-
"""运行价量核心子模型的滚动验证，写入 data/validation.json。

用法：python scripts/run_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec_radar.validation import run_validation  # noqa: E402


def main() -> int:
    result = run_validation(horizon=10, step=20)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:8000])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
