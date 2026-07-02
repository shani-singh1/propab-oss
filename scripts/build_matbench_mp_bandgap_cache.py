#!/usr/bin/env python3
"""One-time build of MP bandgap cache for matbench dielectric rows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.materials_adapter import MaterialsAdapter  # noqa: E402
from propab.domain_adapters.materials_mp_bandgap import (  # noqa: E402
    build_bandgap_cache_from_mp_api,
    build_bandgap_cache_from_mp_gap,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=("mp_gap_fingerprint", "mp_api"),
        default="mp_gap_fingerprint",
        help="mp_gap_fingerprint needs no API key; mp_api uses MP_API_KEY",
    )
    args = p.parse_args()

    adapter = MaterialsAdapter()
    dielectric = adapter.ensure_dataset()
    data_dir = adapter.data_dir

    if args.method == "mp_api":
        out = build_bandgap_cache_from_mp_api(dielectric, data_dir)
    else:
        out = build_bandgap_cache_from_mp_gap(dielectric, data_dir)

    import json

    meta = json.loads(out.read_text(encoding="utf-8")).get("meta", {})
    print(f"Wrote {out}")
    print(f"  matched={meta.get('matched')}/{meta.get('total')} rate={meta.get('match_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
