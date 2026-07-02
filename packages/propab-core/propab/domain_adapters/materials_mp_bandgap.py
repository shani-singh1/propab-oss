"""Cached MP DFT bandgap lookup for matbench dielectric rows."""
from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

_MP_GAP_URL = "https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz"
_CACHE_NAME = "matbench_dielectric_mp_bandgap.json"
_FINGERPRINT_CACHE = "matbench_mp_gap_fingerprints.json.gz"


def bandgap_cache_path(data_dir: Path) -> Path:
    return data_dir / _CACHE_NAME


def _structure_fingerprint(struct: dict[str, Any]) -> tuple[str, int, int, int]:
    """Cheap key for cross-dataset bandgap matching (formula + SG + nsites + volume)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    st = Structure.from_dict(struct)
    sg = int(SpacegroupAnalyzer(st, symprec=0.1).get_space_group_number())
    vol = int(round(float(st.volume)))
    nsites = len(st)
    formula = st.composition.reduced_formula
    return formula, sg, nsites, vol


def load_bandgap_cache(data_dir: Path) -> dict[str, float] | None:
    path = bandgap_cache_path(data_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rows = payload.get("bandgaps_by_index")
    if not isinstance(rows, dict):
        return None
    return {str(k): float(v) for k, v in rows.items() if v is not None}


def save_bandgap_cache(
    data_dir: Path,
    *,
    bandgaps_by_index: dict[int, float | None],
    meta: dict[str, Any] | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = bandgap_cache_path(data_dir)
    payload = {
        "meta": meta or {},
        "bandgaps_by_index": {str(k): v for k, v in bandgaps_by_index.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _urlopen_mp(url: str, *, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "propab-oss/1.0 (materials adapter)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_mp_gap_fingerprints(cache_path: Path) -> dict[tuple[str, int, int, int], float]:
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(_urlopen_mp(_MP_GAP_URL))
    raw = json.loads(gzip.decompress(cache_path.read_bytes()).decode("utf-8"))
    index: dict[tuple[str, int, int, int], float] = {}
    for entry in raw.get("data", []):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        struct, gap = entry[0], entry[1]
        if not isinstance(struct, dict):
            continue
        try:
            key = _structure_fingerprint(struct)
            index[key] = float(gap)
        except Exception:
            continue
    return index


def build_bandgap_cache_from_mp_gap(
    dielectric_path: Path,
    data_dir: Path,
    *,
    mp_gap_cache: Path | None = None,
) -> Path:
    """Match dielectric structures to matbench_mp_gap via composition/SG fingerprint."""
    mp_gap_cache = mp_gap_cache or data_dir / _FINGERPRINT_CACHE
    fp_index = _fetch_mp_gap_fingerprints(mp_gap_cache)
    raw = json.loads(gzip.decompress(dielectric_path.read_bytes()).decode("utf-8"))
    bandgaps: dict[int, float | None] = {}
    matched = 0
    for i, entry in enumerate(raw.get("data", [])):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        struct = entry[0]
        if not isinstance(struct, dict):
            bandgaps[i] = None
            continue
        try:
            key = _structure_fingerprint(struct)
            gap = fp_index.get(key)
            bandgaps[i] = gap
            if gap is not None:
                matched += 1
        except Exception:
            bandgaps[i] = None
    return save_bandgap_cache(
        data_dir,
        bandgaps_by_index=bandgaps,
        meta={
            "source": "matbench_mp_gap_fingerprint_match",
            "matched": matched,
            "total": len(bandgaps),
            "match_rate": round(matched / max(1, len(bandgaps)), 4),
        },
    )


def build_bandgap_cache_from_mp_api(
    dielectric_path: Path,
    data_dir: Path,
    *,
    api_key: str | None = None,
) -> Path:
    """Fetch bandgaps via Materials Project API (structure match)."""
    key = api_key or os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")
    if not key:
        raise ValueError("MP_API_KEY required for API bandgap fetch")
    from mp_api.client import MPRester
    from pymatgen.core import Structure

    raw = json.loads(gzip.decompress(dielectric_path.read_bytes()).decode("utf-8"))
    bandgaps: dict[int, float | None] = {}
    matched = 0
    with MPRester(key) as mpr:
        for i, entry in enumerate(raw.get("data", [])):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            struct = entry[0]
            if not isinstance(struct, dict):
                bandgaps[i] = None
                continue
            try:
                st = Structure.from_dict(struct)
                docs = mpr.summary.search(
                    formula=st.composition.reduced_formula,
                    num_sites=len(st),
                    fields=["material_id", "band_gap"],
                )
                gap = None
                for doc in docs[:5]:
                    bg = getattr(doc, "band_gap", None)
                    if bg is not None:
                        gap = float(bg)
                        break
                bandgaps[i] = gap
                if gap is not None:
                    matched += 1
            except Exception:
                bandgaps[i] = None
    return save_bandgap_cache(
        data_dir,
        bandgaps_by_index=bandgaps,
        meta={
            "source": "materials_project_api",
            "matched": matched,
            "total": len(bandgaps),
            "match_rate": round(matched / max(1, len(bandgaps)), 4),
        },
    )
