"""Fetch the real graph datasets used by graph_invariants + network_diffusion.

These are git-ignored (large-ish binary edge lists) and were previously placed by
hand, which made them a silent single point of failure — a worktree cleanup that
followed a ``data/`` junction once deleted them and errored the graph suites. This
script regenerates them reproducibly:

    python scripts/fetch_graph_datasets.py

- ca-GrQc (collaboration)        — SNAP, direct .txt.gz
- email-Eu-core (communication)  — SNAP, direct .txt.gz
- power-US-Grid (infrastructure) — KONECT opsahl-powergrid bundle, re-emitted as a
  gzipped '#'-commented edge list matching the adapter's reader.

The adapters fail CLOSED (raise) when a file is missing rather than fabricating
synthetic topology, so running this restores real-data operation.
"""
from __future__ import annotations

import gzip
import io
import os
import pathlib
import tarfile
import urllib.request

DEST = pathlib.Path(__file__).resolve().parent.parent / "data" / "v1_candidates"

SNAP = {
    "ca-GrQc.txt.gz": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "email-Eu-core.txt.gz": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
}
KONECT_POWERGRID = "http://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2"
POWERGRID_OUT = "power-US-Grid.txt.gz"


def _fetch(url: str, timeout: float = 90.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "propab-graph-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_snap() -> None:
    for name, url in SNAP.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 1000:
            print(f"[snap] {name}: present ({out.stat().st_size} bytes) — skip")
            continue
        data = _fetch(url)
        out.write_bytes(data)
        print(f"[snap] {name}: fetched {len(data)} bytes")


def fetch_powergrid() -> None:
    out = DEST / POWERGRID_OUT
    if out.exists() and out.stat().st_size > 1000:
        print(f"[konect] {POWERGRID_OUT}: present ({out.stat().st_size} bytes) — skip")
        return
    data = _fetch(KONECT_POWERGRID)
    tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2")
    member = next(n for n in tf.getnames() if os.path.basename(n).startswith("out."))
    raw = tf.extractfile(member).read().decode("utf-8", "ignore")
    edges = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            edges.append((parts[0], parts[1]))
    body = (
        "# US Western States Power Grid (KONECT opsahl-powergrid; "
        "Watts-Strogatz, Nature 393, 1998)\n"
        + "\n".join(f"{a} {b}" for a, b in edges)
        + "\n"
    )
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        fh.write(body)
    print(f"[konect] {POWERGRID_OUT}: {len(edges)} edges -> {out.stat().st_size} bytes")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    fetch_snap()
    fetch_powergrid()
    print("done — real graph datasets present under data/v1_candidates/")


if __name__ == "__main__":
    main()
