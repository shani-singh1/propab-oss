"""Sequence motif / ORF / restriction-site scanning (deterministic, honest).

B1 (domain-capabilities §2a): a ``motif_scan`` tool over the pattern-finding a biologist
repeats across investigations — locating a consensus (IUPAC) motif or a position-weight
matrix, enumerating open reading frames in all six frames, and finding common restriction
sites — using numpy + Bio.Seq, entirely offline.

Honesty by construction (domain-capabilities §0):
  * Every reported hit is a real position in the input: consensus/restriction matches are
    exact IUPAC matches (the caller can re-check the substring), and PWM scores are the
    exact sum of the supplied matrix weights at that offset. Positions are 0-based.
  * ORFs come from an actual start->stop scan of each reading frame with the requested
    codon table; the translated protein is included so the caller can re-verify it.
  * Inputs are validated (non-empty DNA over ACGTN; a well-formed motif / PWM); bad input
    returns a ``validation_error`` and the tool never raises or fabricates a hit.
"""
from __future__ import annotations

import re

from propab.tools.types import ToolError, ToolResult

_DNA_ALPHABET = set("ACGTN")
_IUPAC_ALPHABET = set("ACGTURYSWKMBDHVN")

# IUPAC nucleotide code -> regex character class (U folded to T).
_IUPAC_REGEX = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]", "K": "[GT]", "M": "[AC]",
    "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}

# A small, standard set of common restriction enzymes (palindromic recognition sites).
_RESTRICTION = {
    "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "NotI": "GCGGCCGC",
    "EcoRV": "GATATC", "PstI": "CTGCAG", "SmaI": "CCCGGG", "XhoI": "CTCGAG",
    "SalI": "GTCGAC", "NcoI": "CCATGG", "KpnI": "GGTACC", "SacI": "GAGCTC",
    "SpeI": "ACTAGT", "XbaI": "TCTAGA", "NdeI": "CATATG", "BglII": "AGATCT",
    "NheI": "GCTAGC", "HaeIII": "GGCC", "AluI": "AGCT", "TaqI": "TCGA",
}

_ALLOWED_OPS = frozenset({"consensus", "pwm", "orf", "restriction"})

_MAX_SEQ_LEN = 1_000_000
_MAX_HITS = 10_000

TOOL_SPEC = {
    "name": "motif_scan",
    "domain": "biology",
    "audience": "worker",
    "description": (
        "Deterministic offline scanning of a DNA sequence. op is one of: consensus (find an "
        "IUPAC consensus motif; exact matches, optional both_strands), pwm (score a "
        "position-weight matrix {base:[w0,w1,...]} at every offset, return hits >= threshold "
        "or the top ones), orf (enumerate open reading frames in all six frames with a codon "
        "table, min_aa filter), restriction (find common restriction sites; default a "
        "built-in enzyme set or a supplied 'enzymes' list). Positions are 0-based. Validates "
        "the sequence (ACGTN) and motif/matrix; bad input -> validation_error."
    ),
    "params": {
        "op": {"type": "str", "required": True,
               "description": "consensus | pwm | orf | restriction."},
        "sequence": {"type": "str", "required": True, "description": "DNA sequence over ACGTN."},
        "motif": {"type": "str", "required": False,
                  "description": "consensus: an IUPAC motif string (e.g. 'GAATTC', 'TATAWAW')."},
        "both_strands": {"type": "bool", "required": False,
                         "description": "consensus/restriction: also scan the reverse strand."},
        "pwm": {"type": "dict", "required": False,
                "description": "pwm: {base: [weights...]} with equal-length rows over A,C,G,T."},
        "threshold": {"type": "float", "required": False,
                      "description": "pwm: minimum score for a hit (default: return top 50)."},
        "top_k": {"type": "int", "required": False,
                  "description": "pwm: number of top hits when no threshold (default 50)."},
        "table": {"type": "int", "required": False,
                  "description": "orf: NCBI codon table id (default 1)."},
        "min_aa": {"type": "int", "required": False,
                   "description": "orf: minimum protein length in residues (default 1)."},
        "atg_only": {"type": "bool", "required": False,
                     "description": "orf: require ATG starts (default True) vs all table starts."},
        "enzymes": {"type": "list[str]", "required": False,
                    "description": "restriction: enzyme names to scan (default the built-in set)."},
    },
    "output": {
        "op": "str — the operation performed",
        "matches": "list — consensus/pwm/restriction hits (start, end, strand, matched/score)",
        "orfs": "list — orf: {frame, strand, start, end, length_nt, length_aa, protein, ...}",
        "n_matches": "int — number of hits",
    },
    "example": {
        "params": {"op": "consensus", "sequence": "AAAGAATTCTTT", "motif": "GAATTC"},
        "output": {"op": "consensus", "n_matches": 1,
                   "matches": [{"start": 3, "end": 9, "strand": "+", "matched": "GAATTC"}]},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _clean(seq) -> str:
    return "".join(str(seq).split()).upper()


def _revcomp(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp[b] for b in reversed(seq))


def _iupac_pattern(motif: str) -> re.Pattern:
    return re.compile("".join(_IUPAC_REGEX[c] for c in motif))


def _find_iupac(seq: str, pattern: re.Pattern, width: int) -> list[int]:
    """All 0-based start positions where ``pattern`` matches (overlaps allowed)."""
    starts = []
    for i in range(0, len(seq) - width + 1):
        if pattern.match(seq, i, i + width):
            starts.append(i)
    return starts


def _scan_consensus(seq: str, motif: str, both_strands: bool):
    width = len(motif)
    pattern = _iupac_pattern(motif)
    matches = []
    for start in _find_iupac(seq, pattern, width):
        matches.append({"start": start, "end": start + width, "strand": "+",
                        "matched": seq[start:start + width]})
    if both_strands:
        rc_motif = _revcomp_iupac(motif)
        rc_pattern = _iupac_pattern(rc_motif)
        for start in _find_iupac(seq, rc_pattern, width):
            matches.append({"start": start, "end": start + width, "strand": "-",
                            "matched": seq[start:start + width]})
        matches.sort(key=lambda m: (m["start"], m["strand"]))
    return matches


def _revcomp_iupac(motif: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A", "N": "N",
            "R": "Y", "Y": "R", "S": "S", "W": "W", "K": "M", "M": "K",
            "B": "V", "V": "B", "D": "H", "H": "D"}
    return "".join(comp[c] for c in reversed(motif))


def _scan_pwm(seq: str, pwm: dict, threshold, top_k: int):
    import numpy as np

    bases = ["A", "C", "G", "T"]
    if not isinstance(pwm, dict) or not all(b in pwm for b in bases):
        return None, _validation_error("'pwm' must be a dict with keys A, C, G, T.")
    widths = {len(pwm[b]) for b in bases}
    if len(widths) != 1:
        return None, _validation_error("'pwm' rows (A,C,G,T) must all have equal length.")
    width = widths.pop()
    if width == 0:
        return None, _validation_error("'pwm' must have width >= 1.")
    try:
        mat = np.array([[float(x) for x in pwm[b]] for b in bases], dtype=float)
    except (TypeError, ValueError):
        return None, _validation_error("'pwm' weights must be numbers.")
    idx = {b: i for i, b in enumerate(bases)}
    hits = []
    for i in range(0, len(seq) - width + 1):
        window = seq[i:i + width]
        if any(b not in idx for b in window):  # skip windows containing 'N'
            continue
        score = float(sum(mat[idx[b], k] for k, b in enumerate(window)))
        hits.append((i, score, window))
    if threshold is not None:
        selected = [(i, s, w) for (i, s, w) in hits if s >= threshold]
        selected.sort(key=lambda t: (-t[1], t[0]))
    else:
        selected = sorted(hits, key=lambda t: (-t[1], t[0]))[:top_k]
    matches = [{"start": i, "end": i + width, "strand": "+",
                "score": round(s, 6), "window": w} for (i, s, w) in selected]
    return matches, None


def _find_orfs(seq: str, table_id: int, min_aa: int, atg_only: bool):
    from Bio.Data import CodonTable
    from Bio.Seq import Seq

    table = CodonTable.unambiguous_dna_by_id[table_id]
    stops = set(table.stop_codons)
    starts = {"ATG"} if atg_only else set(table.start_codons)
    n = len(seq)
    orfs = []

    def scan(strand_seq: str, strand: str):
        for frame in range(3):
            i = frame
            orf_start = None
            while i + 3 <= len(strand_seq):
                codon = strand_seq[i:i + 3]
                if orf_start is None:
                    if codon in starts and "N" not in codon:
                        orf_start = i
                elif codon in stops:
                    coding = strand_seq[orf_start:i]  # excludes the stop codon
                    protein = str(Seq(coding).translate(table=table_id, to_stop=False))
                    if len(protein) >= min_aa:
                        # map coordinates back to forward-strand 0-based [start, end)
                        if strand == "+":
                            f_start, f_end = orf_start, i + 3
                        else:
                            f_start, f_end = n - (i + 3), n - orf_start
                        orfs.append({
                            "frame": (frame + 1) if strand == "+" else -(frame + 1),
                            "strand": strand,
                            "start": f_start,
                            "end": f_end,
                            "length_nt": (i + 3) - orf_start,
                            "length_aa": len(protein),
                            "start_codon": strand_seq[orf_start:orf_start + 3],
                            "stop_codon": codon,
                            "protein": protein,
                        })
                    orf_start = None
                i += 3

    scan(seq, "+")
    scan(_revcomp(seq), "-")
    orfs.sort(key=lambda o: (o["start"], o["frame"]))
    return orfs


def motif_scan(
    op: str | None = None,
    sequence=None,
    motif=None,
    both_strands: bool = False,
    pwm=None,
    threshold=None,
    top_k: int = 50,
    table: int = 1,
    min_aa: int = 1,
    atg_only: bool = True,
    enzymes=None,
) -> ToolResult:
    if not op or not isinstance(op, str):
        return _validation_error("Parameter 'op' (string) is required.")
    op = op.strip().lower()
    if op not in _ALLOWED_OPS:
        return _validation_error(f"Unknown op {op!r}; supported: {sorted(_ALLOWED_OPS)}.")

    if sequence is None:
        return _validation_error("Parameter 'sequence' is required.")
    seq = _clean(sequence)
    if not seq:
        return _validation_error("'sequence' is empty after cleaning whitespace.")
    if len(seq) > _MAX_SEQ_LEN:
        return _validation_error(f"sequence length {len(seq)} exceeds the {_MAX_SEQ_LEN} cap.")
    bad = sorted(set(seq) - _DNA_ALPHABET)
    if bad:
        return _validation_error(
            f"'sequence' contains non-DNA characters {bad} (allowed: A,C,G,T,N)."
        )

    try:
        # ---------------------------------------------------------- consensus
        if op == "consensus":
            if not motif:
                return _validation_error("op 'consensus' requires a 'motif' string.")
            m = _clean(motif)
            bad_m = sorted(set(m) - _IUPAC_ALPHABET)
            if bad_m:
                return _validation_error(f"'motif' has invalid IUPAC codes {bad_m}.")
            if len(m) > len(seq):
                return _validation_error("'motif' is longer than the sequence.")
            matches = _scan_consensus(seq, m, bool(both_strands))
            return ToolResult(success=True, output={
                "op": op, "motif": m, "n_matches": len(matches),
                "matches": matches[:_MAX_HITS],
                "method": "exact IUPAC match (overlaps allowed)",
            })

        # ---------------------------------------------------------- pwm
        if op == "pwm":
            if pwm is None:
                return _validation_error("op 'pwm' requires a 'pwm' matrix.")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
                return _validation_error("'top_k' must be a positive integer.")
            if threshold is not None:
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    return _validation_error("'threshold' must be a number.")
            matches, err = _scan_pwm(seq, pwm, threshold, top_k)
            if err:
                return err
            return ToolResult(success=True, output={
                "op": op, "n_matches": len(matches), "matches": matches[:_MAX_HITS],
                "threshold": threshold,
                "method": "sum of PWM weights per offset (windows with N skipped)",
            })

        # ---------------------------------------------------------- orf
        if op == "orf":
            from Bio.Data import CodonTable

            if (not isinstance(table, int) or isinstance(table, bool)
                    or table not in CodonTable.unambiguous_dna_by_id):
                return _validation_error(f"'table' must be a valid NCBI codon table id (got {table!r}).")
            if not isinstance(min_aa, int) or isinstance(min_aa, bool) or min_aa < 0:
                return _validation_error("'min_aa' must be a non-negative integer.")
            orfs = _find_orfs(seq, table, min_aa, bool(atg_only))
            return ToolResult(success=True, output={
                "op": op, "n_orfs": len(orfs), "orfs": orfs[:_MAX_HITS],
                "table": table, "min_aa": min_aa, "atg_only": bool(atg_only),
                "method": "start->stop scan over all six reading frames",
            })

        # ---------------------------------------------------------- restriction
        if op == "restriction":
            if enzymes is not None:
                if not isinstance(enzymes, (list, tuple)):
                    return _validation_error("'enzymes' must be a list of enzyme names.")
                unknown = [e for e in enzymes if e not in _RESTRICTION]
                if unknown:
                    return _validation_error(
                        f"Unknown enzyme(s) {unknown}; known: {sorted(_RESTRICTION)}."
                    )
                enzyme_set = {e: _RESTRICTION[e] for e in enzymes}
            else:
                enzyme_set = dict(_RESTRICTION)
            matches = []
            for name, site in enzyme_set.items():
                width = len(site)
                pattern = _iupac_pattern(site)
                for start in _find_iupac(seq, pattern, width):
                    matches.append({"enzyme": name, "site": site, "start": start,
                                    "end": start + width, "strand": "+"})
                if both_strands:
                    rc_site = _revcomp_iupac(site)
                    if rc_site != site:  # non-palindromic
                        for start in _find_iupac(seq, _iupac_pattern(rc_site), width):
                            matches.append({"enzyme": name, "site": site, "start": start,
                                            "end": start + width, "strand": "-"})
            matches.sort(key=lambda m: (m["start"], m["enzyme"]))
            return ToolResult(success=True, output={
                "op": op, "n_matches": len(matches), "matches": matches[:_MAX_HITS],
                "enzymes_scanned": sorted(enzyme_set),
                "method": "exact IUPAC match of recognition sites",
            })

        return _validation_error(f"Unhandled op {op!r}.")

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"motif_scan op '{op}' failed: {exc}")
