# -*- coding: utf-8 -*-
"""命令行扫描入口。

示例：
    python scripts/scan.py --n 12 --depth 420
    python scripts/scan.py --n-potential 15 --n-active 10 --risk defensive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from spec_radar.config import ScanSettings, load_macro_playbook, load_model_config  # noqa: E402
from spec_radar.pipeline import run_scan  # noqa: E402

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.unicode.east_asian_width", True)


def main() -> int:
    parser = argparse.ArgumentParser(description="A股双轨投机雷达 · 本地命令行")
    parser.add_argument("--n", type=int, default=12, help="两个榜单默认 N")
    parser.add_argument("--n-potential", type=int, default=None)
    parser.add_argument("--n-active", type=int, default=None)
    parser.add_argument("--depth", type=int, default=300, choices=[200, 300, 500, 650], help="每个模型的粗筛池大小")
    parser.add_argument("--risk", default="auto", choices=["auto", "offensive", "defensive"])
    parser.add_argument("--sort", default="profitability", choices=["profitability", "speculation"],
                        help="榜单排序：盈利能力优先或投机综合分优先")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD，默认最近已收盘交易日")
    parser.add_argument("--outdir", default=str(ROOT / "reports"))
    parser.add_argument("--no-cache", action="store_true", help="强制绕过磁盘缓存重新抓取")
    args = parser.parse_args()
    n_pot = args.n_potential or args.n
    n_act = args.n_active or args.n
    settings = ScanSettings(
        n_potential=n_pot, n_active=n_act, scan_potential=args.depth, scan_active=args.depth,
        risk_mode=args.risk, sort_mode=args.sort, as_of=args.as_of,
    )
    model_cfg = load_model_config()
    playbook = load_macro_playbook()
    result = run_scan(
        settings, model_cfg=model_cfg, playbook=playbook,
        progress=lambda stage, frac, detail: print(f"[{stage:4s}] {frac:>5.0%} {detail}", flush=True),
        force=args.no_cache,
    )
    meta = result["meta"]
    print("\n" + "=" * 100)
    print(f"信号日 {meta['signal_date']} | 快照 {meta['snapshot_time']} | 市场 {meta['macro_label']} | 耗时 {meta['elapsed_sec']}s")
    for key, label, n in (("potential", "潜在投机榜", n_pot), ("active", "活跃投机榜", n_act)):
        df = result[key]
        print("\n" + "-" * 100)
        print(f"{label} · 共 {len(df)} 只合格，显示前 {min(n, len(df))} 只")
        if df.empty:
            print("（无）")
            continue
        show = df.head(n)[
            ["rank", "code", "name", "score", "confidence", "risk_level", "industry", "theme_label",
             "ret20", "ret60", "float_mcap_yi", "amount20_yi", "turnover20_pct", "risk_flags_text"]
        ].copy()
        for col in ("ret20", "ret60"):
            show[col] = (show[col] * 100).map(lambda x: f"{x:.1f}%")
        show["turnover20_pct"] = show["turnover20_pct"].map(lambda x: f"{x:.1f}%")
        print(show.to_string(index=False))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{key}_rank_{meta['signal_date']}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"已保存：{path}")
    if meta.get("warnings"):
        print("\n数据提示：")
        for w in meta["warnings"]:
            print("-", w)
    print("\n⚠️ 仅供研究与风险识别，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
