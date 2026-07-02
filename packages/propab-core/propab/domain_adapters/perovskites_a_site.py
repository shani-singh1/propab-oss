"""A-site chemistry grouping for matbench perovskites (ABX3-like)."""
from __future__ import annotations

from pymatgen.core import Structure

from propab.domain_adapters.materials_element_data import element_props

_ANIONS = frozenset({"O", "N", "F", "Cl", "Br", "I", "S", "Se", "Te"})


def a_site_element(struct: Structure) -> str:
    """
    Identify A-site proxy as the most electropositive cation (lowest Pauling EN)
    among species that are not dominant anions in the unit cell.
    """
    comp = struct.composition
    cations: list[tuple[str, float, float]] = []
    for el, amt in comp.items():
        sym = el.symbol
        frac = float(amt / comp.num_atoms)
        if sym in _ANIONS and frac >= 0.35:
            continue
        _z, _mass, en = element_props(sym)
        cations.append((sym, en, frac))
    if not cations:
        return "unknown"
    # Prefer electropositive (low EN); tie-break by smaller fractional occupancy (A is 1/5 in ABX3).
    cations.sort(key=lambda row: (row[1], row[2]))
    return cations[0][0]


def a_site_group(struct: Structure, *, min_count_bucket: int = 0) -> str:
    """Return A-site element symbol (family key for LOFO)."""
    return a_site_element(struct)
