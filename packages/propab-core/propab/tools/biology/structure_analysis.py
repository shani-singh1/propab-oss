"""PDB / mmCIF structure analysis (parsed from real coordinates, honest).

B1 (domain-capabilities §2a): a ``structure_analysis`` tool over the repetitive structural
bookkeeping — parse a PDB or mmCIF structure supplied as text and report its chains,
residue counts, a secondary-structure summary where the file carries it, and inter-residue
distances / contacts — via ``Bio.PDB`` reading from an in-memory ``StringIO`` (no file I/O,
no network).

Honesty by construction (domain-capabilities §0):
  * Every number is computed from the parsed structure: chain/residue/atom counts come
    from the object model, distances from the real Cartesian coordinates. Nothing is
    inferred or guessed.
  * Secondary structure is read from the file's own HELIX/SHEET (PDB) or _struct_conf /
    _struct_sheet_range (mmCIF) annotation records ONLY. DSSP needs an external binary that
    is not available offline, so when the file carries no annotation the tool reports
    ``available: false`` rather than fabricating helices/strands.
  * A structure that fails to parse returns a clear ``execution_error``; empty/whitespace
    input returns a ``validation_error``. The tool never raises to the caller.
"""
from __future__ import annotations

from io import StringIO

from propab.tools.types import ToolError, ToolResult

# Cap the residue count for the all-vs-all contact map so a huge structure cannot hang.
_MAX_CONTACT_RESIDUES = 4_000
_MAX_TEXT_LEN = 50_000_000

TOOL_SPEC = {
    "name": "structure_analysis",
    "domain": "biology",
    "audience": "worker",
    "description": (
        "Parse a molecular structure supplied as text (PDB or mmCIF) via Bio.PDB and report "
        "chains, per-chain residue counts, total polymer/hetero/water residues, atom count, "
        "a secondary-structure summary read from the file's HELIX/SHEET (PDB) or _struct_conf "
        "(mmCIF) records where present, and inter-residue CA-CA contacts within "
        "contact_threshold Angstroms (sequence separation > min_seq_sep). Optionally pass "
        "'pairs' [[chain,resseq,chain,resseq], ...] to measure specific inter-residue "
        "distances. Format is 'pdb', 'cif', or 'auto'. Empty input -> validation_error; an "
        "unparseable structure -> execution_error (never a guessed result)."
    ),
    "params": {
        "structure": {"type": "str", "required": True,
                      "description": "The structure file contents as text (PDB or mmCIF)."},
        "format": {"type": "str", "required": False,
                   "description": "'pdb', 'cif', or 'auto' (default)."},
        "contact_threshold": {"type": "float", "required": False,
                              "description": "CA-CA contact distance in Angstroms (default 8.0)."},
        "min_seq_sep": {"type": "int", "required": False,
                        "description": "Min residue separation for same-chain contacts (default 2)."},
        "pairs": {"type": "list", "required": False,
                  "description": "Specific residue pairs [[chain,resseq,chain,resseq], ...] "
                                 "to measure CA-CA distance for."},
    },
    "output": {
        "format": "str — pdb | cif (as parsed)",
        "models": "int — number of models",
        "chains": "list[dict] — {id, residues, hetero_residues, waters, atoms}",
        "n_chains": "int",
        "total_residues": "int — polymer residues across all chains",
        "total_atoms": "int",
        "secondary_structure": "dict — {available, source, helices, strands, ...}",
        "contacts": "dict — {threshold, min_seq_sep, count, sample}",
        "pair_distances": "list — requested pair measurements (or 'error' per pair)",
    },
    "example": {
        "params": {"structure": "ATOM ... (PDB text)", "format": "pdb"},
        "output": {"format": "pdb", "n_chains": 1, "total_residues": 2},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _detect_format(text: str) -> str:
    """Heuristic PDB vs mmCIF detection from the text contents."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("data_") or s.startswith("loop_") or s.startswith("_atom_site."):
            return "cif"
        if s[:6].strip() in ("ATOM", "HETATM", "HEADER", "HELIX", "SHEET", "SEQRES", "CRYST1"):
            return "pdb"
    # default: mmCIF files usually start with data_; otherwise assume PDB
    return "pdb"


def _parse_pdb_secondary_structure(text: str) -> dict:
    """Count HELIX/SHEET records (and residues spanned) directly from PDB header lines."""
    helices = strands = 0
    helix_residues = strand_residues = 0
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "HELIX":
            helices += 1
            try:  # PDB fixed columns: initSeqNum 22-25, endSeqNum 34-37 (1-based)
                init = int(line[21:25])
                end = int(line[33:37])
                if end >= init:
                    helix_residues += end - init + 1
            except (ValueError, IndexError):
                pass
        elif rec == "SHEET":
            strands += 1
            try:  # SHEET: initSeqNum 23-26, endSeqNum 34-37 (1-based)
                init = int(line[22:26])
                end = int(line[33:37])
                if end >= init:
                    strand_residues += end - init + 1
            except (ValueError, IndexError):
                pass
    if helices == 0 and strands == 0:
        return {"available": False, "source": "PDB HELIX/SHEET records",
                "note": "no HELIX/SHEET records in the file; DSSP unavailable offline"}
    return {"available": True, "source": "PDB HELIX/SHEET records",
            "helices": helices, "helix_residues": helix_residues,
            "strands": strands, "strand_residues": strand_residues}


def _parse_cif_secondary_structure(text: str) -> dict:
    """Best-effort helix/strand counts from mmCIF _struct_conf / _struct_sheet_range."""
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict

        d = MMCIF2Dict(StringIO(text))
    except Exception:  # noqa: BLE001
        return {"available": False, "source": "mmCIF _struct_conf",
                "note": "could not read mmCIF annotation records"}

    def _count(key: str) -> int:
        val = d.get(key)
        if val is None:
            return 0
        return len(val) if isinstance(val, list) else 1

    helices = _count("_struct_conf.id")
    strands = _count("_struct_sheet_range.id")
    if helices == 0 and strands == 0:
        return {"available": False, "source": "mmCIF _struct_conf / _struct_sheet_range",
                "note": "no secondary-structure annotation records in the file"}
    return {"available": True, "source": "mmCIF _struct_conf / _struct_sheet_range",
            "helices": helices, "strands": strands}


def _classify_residue(residue) -> str:
    """'polymer', 'water', or 'hetero' from the residue hetero flag."""
    hetflag = residue.id[0]
    if hetflag == "W":
        return "water"
    if hetflag != " ":
        return "hetero"
    return "polymer"


def structure_analysis(
    structure=None,
    format: str = "auto",
    contact_threshold: float = 8.0,
    min_seq_sep: int = 2,
    pairs=None,
) -> ToolResult:
    if structure is None or not isinstance(structure, str) or not structure.strip():
        return _validation_error("Parameter 'structure' must be non-empty structure text.")
    if len(structure) > _MAX_TEXT_LEN:
        return _validation_error(f"structure text exceeds the {_MAX_TEXT_LEN}-char cap.")

    fmt = str(format).strip().lower()
    if fmt not in ("pdb", "cif", "mmcif", "auto"):
        return _validation_error("'format' must be 'pdb', 'cif', or 'auto'.")
    if fmt == "mmcif":
        fmt = "cif"
    if fmt == "auto":
        fmt = _detect_format(structure)

    try:
        contact_threshold = float(contact_threshold)
    except (TypeError, ValueError):
        return _validation_error("'contact_threshold' must be a number.")
    if contact_threshold <= 0:
        return _validation_error("'contact_threshold' must be > 0.")
    if not isinstance(min_seq_sep, int) or isinstance(min_seq_sep, bool) or min_seq_sep < 0:
        return _validation_error("'min_seq_sep' must be a non-negative integer.")

    try:
        import warnings

        import numpy as np

        if fmt == "cif":
            from Bio.PDB import MMCIFParser

            parser = MMCIFParser(QUIET=True)
        else:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                struct = parser.get_structure("input", StringIO(structure))
            except Exception as exc:  # noqa: BLE001
                return _execution_error(
                    f"failed to parse {fmt} structure: {exc}. "
                    f"If the format was auto-detected wrongly, pass 'format' explicitly."
                )

        models = list(struct)
        if not models:
            return _execution_error("parsed structure contains no models.")
        model = models[0]

        # ---- chain / residue / atom census over the first model ----
        chains_out = []
        total_polymer = total_atoms = 0
        ca_records = []  # (chain_id, resseq, np.array coord) for polymer residues with a CA
        for chain in model:
            polymer = hetero = waters = atoms = 0
            for residue in chain:
                kind = _classify_residue(residue)
                atoms += sum(1 for _ in residue)
                if kind == "polymer":
                    polymer += 1
                    if "CA" in residue:
                        ca_records.append((chain.id, residue.id[1],
                                           np.asarray(residue["CA"].coord, dtype=float)))
                elif kind == "water":
                    waters += 1
                else:
                    hetero += 1
            chains_out.append({"id": chain.id, "residues": polymer,
                               "hetero_residues": hetero, "waters": waters, "atoms": atoms})
            total_polymer += polymer
            total_atoms += atoms

        # ---- secondary structure (from the file's own annotation records only) ----
        if fmt == "cif":
            ss = _parse_cif_secondary_structure(structure)
        else:
            ss = _parse_pdb_secondary_structure(structure)

        # ---- inter-residue CA-CA contacts ----
        if len(ca_records) > _MAX_CONTACT_RESIDUES:
            contacts = {"threshold": contact_threshold, "min_seq_sep": min_seq_sep,
                        "computed": False,
                        "note": f"{len(ca_records)} CA residues exceed the "
                                f"{_MAX_CONTACT_RESIDUES} cap; contact map skipped."}
        else:
            sample = []
            count = 0
            for i in range(len(ca_records)):
                ci, ri, coord_i = ca_records[i]
                for j in range(i + 1, len(ca_records)):
                    cj, rj, coord_j = ca_records[j]
                    if ci == cj and abs(ri - rj) <= min_seq_sep:
                        continue
                    dist = float(np.linalg.norm(coord_i - coord_j))
                    if dist <= contact_threshold:
                        count += 1
                        if len(sample) < 50:
                            sample.append({"chain_a": ci, "res_a": ri,
                                           "chain_b": cj, "res_b": rj,
                                           "distance": round(dist, 3)})
            contacts = {"threshold": contact_threshold, "min_seq_sep": min_seq_sep,
                        "computed": True, "count": count, "sample": sample,
                        "method": "CA-CA Euclidean distance"}

        # ---- specific requested pair distances ----
        pair_distances = []
        if pairs is not None:
            if not isinstance(pairs, (list, tuple)):
                return _validation_error("'pairs' must be a list of [chain,resseq,chain,resseq].")
            for p in pairs:
                entry = {"request": p}
                try:
                    ch_a, res_a, ch_b, res_b = p[0], int(p[1]), p[2], int(p[3])
                    atom_a = model[ch_a][(" ", res_a, " ")]["CA"]
                    atom_b = model[ch_b][(" ", res_b, " ")]["CA"]
                    entry["distance"] = round(
                        float(np.linalg.norm(np.asarray(atom_a.coord) - np.asarray(atom_b.coord))), 3)
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"could not measure: {exc}"
                pair_distances.append(entry)

        return ToolResult(success=True, output={
            "format": fmt,
            "models": len(models),
            "chains": chains_out,
            "n_chains": len(chains_out),
            "total_residues": total_polymer,
            "total_atoms": total_atoms,
            "secondary_structure": ss,
            "contacts": contacts,
            "pair_distances": pair_distances,
        })

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"structure_analysis failed: {exc}")
