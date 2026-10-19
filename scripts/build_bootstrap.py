# -*- coding: utf-8 -*-
"""生成 data/bootstrap 离线灾备包。

用法：
    python scripts/build_bootstrap.py [--depth 420]
建议在每次代码/配置更新、准备部署前运行一次，然后提交 data/bootstrap。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from spec_radar.bootstrap import (  # noqa: E402
    BOARD_KLINE_PATH,
    BOOTSTRAP_DIR,
    CONCEPT_PATH,
    INDUSTRY_PATH,
    MARKET_KLINE_PATH,
    META_PATH,
    SNAPSHOT_PATH,
    STOCK_KLINE_PATH,
)
from spec_radar.config import ScanSettings  # noqa: E402
from spec_radar.datasource import MarketClient, determine_signal_date  # noqa: E402
from spec_radar.pipeline import run_scan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=420)
    args = parser.parse_args()
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    client = MarketClient(workers=10)
    signal, probe = determine_signal_date(client)
    print(f"signal date: {signal}")
    snapshot_bundle = client.get_snapshot()
    boards = client.get_boards()
    settings = ScanSettings(n_potential=12, n_active=12, scan_potential=args.depth, scan_active=args.depth)
    result = run_scan(settings, progress=lambda stage, frac, detail: print(f"[{stage}] {frac:.0%} {detail}"))
    panel = result["kline"]
    board_panel = result.get("board_panel", pd.DataFrame())
    if board_panel.empty:
        # 从本机已有缓存里找最近一次的板块指数面板作为兜底。
        import pickle
        from pathlib import Path as _Path
        candidates = sorted((_Path("data/cache")).glob("kline_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                with path.open("rb") as fh:
                    value = pickle.load(fh)["value"]
                if not value.empty and str(value["code"].iloc[0]).startswith("90."):
                    board_panel = value
                    break
            except Exception:
                continue
    context = client.get_market_context_panel(signal, lmt=320)
    snapshot = snapshot_bundle.df.copy()
    snapshot["concepts"] = snapshot["concepts"].map(lambda xs: "|".join(xs) if isinstance(xs, list) else str(xs))
    snapshot.to_parquet(SNAPSHOT_PATH, index=False, compression="zstd")
    boards.industry.to_parquet(INDUSTRY_PATH, index=False, compression="zstd")
    boards.concept.to_parquet(CONCEPT_PATH, index=False, compression="zstd")
    panel.to_parquet(STOCK_KLINE_PATH, index=False, compression="zstd")
    if not board_panel.empty:
        board_panel.to_parquet(BOARD_KLINE_PATH, index=False, compression="zstd")
    context.frame.to_parquet(MARKET_KLINE_PATH, index=False, compression="zstd")
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": str(signal),
        "snapshot_time": snapshot_bundle.fetched_at.strftime("%Y-%m-%d %H:%M"),
        "snapshot_stale": bool(snapshot_bundle.stale),
        "universe": int(len(snapshot)),
        "scanned_codes": int(panel["code"].nunique()) if not panel.empty else 0,
        "kline_rows": int(len(panel)),
        "depth": args.depth,
        "source": "东方财富公开行情 / 腾讯、新浪备用",
        "note": "仅供接口故障时预览，不是实时行情；请以程序实际抓取与数据基准日为准。",
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("bootstrap files:")
    for path in [SNAPSHOT_PATH, INDUSTRY_PATH, CONCEPT_PATH, STOCK_KLINE_PATH, BOARD_KLINE_PATH, MARKET_KLINE_PATH, META_PATH]:
        print(path.name, f"{path.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
