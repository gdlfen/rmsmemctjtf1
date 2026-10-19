# -*- coding: utf-8 -*-
"""训练概率/目标价/起爆窗口参数。

用法：python scripts/train_forecast.py
依赖 data/bootstrap 的候选股历史面板。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec_radar.forecast import train_forecast  # noqa: E402


def main() -> int:
    result = train_forecast(step=10)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
