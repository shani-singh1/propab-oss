"""Derive crystal system / space group from pymatgen structure dicts."""
from __future__ import annotations

from typing import Any

_CRYSTAL_SYSTEM_FROM_SG: tuple[tuple[int, int, str], ...] = (
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
)


def crystal_system_from_space_group_number(sg_number: int) -> str:
    n = int(sg_number)
    for lo, hi, name in _CRYSTAL_SYSTEM_FROM_SG:
        if lo <= n <= hi:
            return name
    return "unknown"


def symmetry_from_structure_dict(struct: dict[str, Any], *, symprec: float = 0.1) -> dict[str, Any]:
    """
    Return crystal_system, space_group_number, space_group_symbol from a matbench structure dict.

    Uses pymatgen SpacegroupAnalyzer on Structure.from_dict — matbench entries are
    already pymatgen-serialized structures.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = Structure.from_dict(struct)
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symbol, number = sga.get_space_group_symbol(), int(sga.get_space_group_number())
    return {
        "crystal_system": sga.get_crystal_system(),
        "space_group_number": number,
        "space_group_symbol": symbol,
        "crystal_system_from_sg": crystal_system_from_space_group_number(number),
    }
