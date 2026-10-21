# -*- coding: utf-8 -*-
"""生成三浪/第一浪历史回测报告。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec_radar.wave_backtest import run_backtest  # noqa: E402


def main() -> None:
    out = run_backtest(horizon=10, step=5, top_n=10, cost=0.002)
    path = ROOT / "data" / "wave_backtest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out.get("results", out), ensure_ascii=False, indent=2))
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
