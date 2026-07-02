#!/usr/bin/env python3
"""
Build a V1 literature prior (standalone side experiment — see scripts/v1_literature_prior/).

Example:
  python scripts/build_v1_literature_prior.py --domain materials
  python scripts/build_v1_literature_prior.py --domain materials --max-papers 25 --out artifacts/v1_literature_prior/materials_dielectric.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from scripts.v1_literature_prior.build import (  # noqa: E402
    build_literature_prior,
    prior_for_campaign_loop,
    write_prior_artifact,
)
from scripts.v1_literature_prior.domains import DEFAULT_DOMAIN  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "v1_literature_prior" / "materials_dielectric.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default=DEFAULT_DOMAIN)
    p.add_argument("--question", default=None, help="Override default research question")
    p.add_argument("--max-papers", type=int, default=30)
    p.add_argument("--model", default=None, help="LLM model (default: LLM_MODEL env or gpt-4o)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--campaign-prior-out",
        default=None,
        help="Also write Propab-shaped prior_json (for manual injection / campaign 2)",
    )
    args = p.parse_args()

    print(f"Fetching papers for domain={args.domain} (max {args.max_papers})...")
    payload = build_literature_prior(
        domain=args.domain,
        question=args.question,
        max_papers=args.max_papers,
        model=args.model,
    )
    out = write_prior_artifact(payload, Path(args.out))
    print(f"Wrote {out}")
    print(
        f"  papers={payload['meta']['paper_count']} "
        f"facts={len(payload['prior'].get('established_facts', []))} "
        f"contested={len(payload['prior'].get('contested_claims', []))} "
        f"gaps={len(payload['prior'].get('open_gaps', []))} "
        f"method={payload['prior'].get('extraction_method')}"
    )

    if args.campaign_prior_out:
        slim = prior_for_campaign_loop(payload)
        cp = Path(args.campaign_prior_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote campaign prior_json shape -> {cp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
