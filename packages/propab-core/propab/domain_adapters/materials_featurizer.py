"""Derive literature-backed descriptors from matbench pymatgen-style structure dicts."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from propab.domain_adapters.materials_element_data import element_props, principal_quantum_n

_AMU_PER_ANG3_TO_G_CM3 = 1.66053906660
_COORD_CUTOFF_ANG = 3.5


def lattice_volume_ang3(lattice: dict[str, Any] | None) -> float | None:
    if not isinstance(lattice, dict):
        return None
    matrix = lattice.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 3:
        try:
            m = np.array(matrix, dtype=float)
            vol = abs(float(np.linalg.det(m)))
            return vol if vol > 0 else None
        except (TypeError, ValueError):
            pass
    vol = lattice.get("volume")
    if isinstance(vol, (int, float)) and float(vol) > 0:
        return float(vol)
    return None


def _species_rows(site: dict[str, Any]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for sp in site.get("species") or []:
        if not isinstance(sp, dict):
            continue
        el = str(sp.get("element") or "").strip()
        if not el:
            continue
        occu = float(sp.get("occu") or 1.0)
        rows.append((el, occu))
    return rows


def featurize_structure(struct: dict[str, Any]) -> dict[str, float]:
    sites = struct.get("sites") or []
    z_vals: list[float] = []
    n_vals: list[float] = []
    masses: list[float] = []
    en_vals: list[float] = []
    elements: set[str] = set()
    total_mass = 0.0

    for site in sites:
        if not isinstance(site, dict):
            continue
        for el, occu in _species_rows(site):
            elements.add(el)
            z, mass, en = element_props(el)
            n_vals.append(float(principal_quantum_n(z)))
            z_vals.extend([float(z)] * max(1, int(round(occu))))
            masses.append(mass)
            en_vals.append(en)
            total_mass += mass * occu

    n_sites = len(sites)
    n_elements = len(elements)
    mean_z = float(np.mean(z_vals)) if z_vals else 0.0
    std_z = float(np.std(z_vals)) if len(z_vals) > 1 else 0.0
    std_principal_quantum_n = float(np.std(n_vals)) if len(n_vals) > 1 else 0.0
    mean_atomic_mass = float(np.mean(masses)) if masses else 0.0
    mean_electronegativity = float(np.mean(en_vals)) if en_vals else 0.0

    # Mean |χ_i - χ_j| across unique element pairs (ionic character proxy).
    en_by_el = {el: element_props(el)[2] for el in elements}
    el_list = sorted(elements)
    ionic_diffs: list[float] = []
    for i, a in enumerate(el_list):
        for b in el_list[i + 1 :]:
            ionic_diffs.append(abs(en_by_el[a] - en_by_el[b]))
    mean_ionicity = float(np.mean(ionic_diffs)) if ionic_diffs else 0.0

    vol = lattice_volume_ang3(struct.get("lattice"))
    mass_density = (
        (total_mass / vol) * _AMU_PER_ANG3_TO_G_CM3 if vol and vol > 0 and total_mass > 0 else 0.0
    )

    coords = []
    for site in sites:
        xyz = site.get("xyz")
        if isinstance(xyz, list) and len(xyz) == 3:
            coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    mean_coordination = 0.0
    if len(coords) >= 2:
        arr = np.array(coords, dtype=float)
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        neighbor_counts = (dists < _COORD_CUTOFF_ANG).sum(axis=1)
        mean_coordination = float(np.mean(neighbor_counts))

    return {
        "n_sites": float(n_sites),
        "n_elements": float(n_elements),
        "mean_Z": mean_z,
        "std_Z": std_z,
        "std_principal_quantum_n": std_principal_quantum_n,
        "mean_atomic_mass": mean_atomic_mass,
        "mass_density": mass_density,
        "mean_electronegativity": mean_electronegativity,
        "mean_ionicity": mean_ionicity,
        "mean_coordination": mean_coordination,
    }
