"""Pairwise + simple multiple sequence alignment (deterministic, honest).

B1 (domain-capabilities §2a): a ``sequence_align`` tool over the repetitive alignment
work a molecular biologist runs across thousands of investigations — global/local
pairwise alignment and a simple progressive multiple alignment of DNA or protein
sequences, returning the alignment itself, the score, and a % identity computed from the
real alignment.

Honesty by construction (domain-capabilities §0):
  * The alignment, score and identities come straight from ``Bio.Align.PairwiseAligner``
    (Needleman-Wunsch / Smith-Waterman). ``percent_identity`` is ``identities /
    alignment_length`` counted from the returned alignment columns — never a guessed or
    self-reported number. A hand count on any tiny example reproduces it exactly.
  * The multiple alignment is a deterministic center-star progressive merge of real
    pairwise alignments. The tool independently re-checks that every row, with gaps
    removed, equals its original input sequence, and reports ``consistent`` — it never
    emits a fabricated MSA.
  * Inputs are validated: sequences must be non-empty and over a valid DNA/protein
    alphabet; bad input returns a ``validation_error`` and the tool never raises.
  * The scoring scheme + algorithm + parameters used are reported with every result.
"""
from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

# Valid alphabets (uppercase). DNA/RNA + N; protein = 20 aa + B,Z,X (ambiguity) and *.
_DNA_ALPHABET = set("ACGTUN")
_PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBZX*")

# Guardrails so a pathological input cannot hang the O(n*m) aligner.
_MAX_SEQ_LEN = 20_000
_MAX_SEQUENCES = 50

TOOL_SPEC = {
    "name": "sequence_align",
    "domain": "biology",
    "audience": "worker",
    "description": (
        "Deterministic pairwise (global/local) and simple multiple alignment of DNA or "
        "protein sequences via Bio.Align.PairwiseAligner. Pass 'sequences' (a list; 2 for "
        "pairwise, >2 for a center-star multiple alignment) or 'seq_a'/'seq_b'. mode is "
        "'global' (Needleman-Wunsch) or 'local' (Smith-Waterman). seq_type is 'dna', "
        "'protein' or 'auto'. Returns the aligned rows, the alignment score, identities and "
        "percent_identity (identities / alignment columns, computed from the real "
        "alignment), plus the scoring scheme used. Multiple alignments carry a 'consistent' "
        "flag re-verifying each row de-gaps to its input."
    ),
    "params": {
        "sequences": {"type": "list[str]", "required": False,
                       "description": "List of sequences (2 => pairwise, >2 => multiple)."},
        "seq_a": {"type": "str", "required": False,
                  "description": "First sequence (pairwise convenience if 'sequences' absent)."},
        "seq_b": {"type": "str", "required": False,
                  "description": "Second sequence (pairwise convenience if 'sequences' absent)."},
        "mode": {"type": "str", "required": False,
                 "description": "'global' (default) or 'local'."},
        "seq_type": {"type": "str", "required": False,
                     "description": "'dna', 'protein', or 'auto' (default)."},
    },
    "output": {
        "kind": "str — 'pairwise' or 'multiple'",
        "mode": "str — global | local",
        "seq_type": "str — dna | protein",
        "algorithm": "str — the alignment algorithm used",
        "scoring": "dict — match/mismatch or substitution matrix + gap penalties",
        "score": "float — the alignment score",
        "alignment": "list[str] — aligned rows (gaps as '-')",
        "identities": "int — number of identical aligned positions",
        "alignment_length": "int — number of alignment columns",
        "percent_identity": "float — 100 * identities / alignment_length",
        "consistent": "bool — multiple only: every row de-gaps to its input",
    },
    "example": {
        "params": {"seq_a": "ACGTACGT", "seq_b": "ACGTTCGT", "mode": "global",
                   "seq_type": "dna"},
        "output": {
            "kind": "pairwise", "mode": "global", "seq_type": "dna",
            "score": 13.0, "identities": 7, "alignment_length": 8,
            "percent_identity": 87.5,
        },
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _clean(seq) -> str:
    """Uppercase and strip whitespace from a candidate sequence string."""
    return "".join(str(seq).split()).upper()


def _detect_type(seqs: list[str]) -> str | None:
    """Auto-detect DNA vs protein from the union of characters; None if neither fits."""
    chars = set().union(*[set(s) for s in seqs])
    if chars <= _DNA_ALPHABET:
        return "dna"
    if chars <= _PROTEIN_ALPHABET:
        return "protein"
    return None


def _validate_alphabet(seqs: list[str], seq_type: str):
    alphabet = _DNA_ALPHABET if seq_type == "dna" else _PROTEIN_ALPHABET
    for i, s in enumerate(seqs):
        bad = sorted(set(s) - alphabet)
        if bad:
            return _validation_error(
                f"sequence[{i}] contains characters {bad} not valid for seq_type "
                f"'{seq_type}' (allowed: {''.join(sorted(alphabet))})."
            )
    return None


def _build_aligner(mode: str, seq_type: str):
    """Construct a deterministic PairwiseAligner and return (aligner, scoring dict)."""
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = mode
    if seq_type == "protein":
        from Bio.Align import substitution_matrices

        matrix = substitution_matrices.load("BLOSUM62")
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        scoring = {"substitution_matrix": "BLOSUM62",
                   "open_gap_score": -11, "extend_gap_score": -1}
    else:
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -2
        scoring = {"match_score": 2, "mismatch_score": -1,
                   "open_gap_score": -5, "extend_gap_score": -2}
    return aligner, scoring


def _pairwise(aligner, a: str, b: str):
    """Return (score, row_a, row_b, identities, alignment_length) for the best alignment."""
    alignments = aligner.align(a, b)
    best = alignments[0]
    row_a, row_b = str(best[0]), str(best[1])
    counts = best.counts()
    identities = int(counts.identities)
    aln_len = len(row_a)
    return float(best.score), row_a, row_b, identities, aln_len


def _center_star(aligner, seqs: list[str]):
    """Deterministic center-star progressive multiple alignment.

    1. pick the center = sequence maximising the sum of pairwise scores to the others
       (ties -> lowest index);
    2. align every other sequence to the center;
    3. merge the pairwise alignments into one MSA, propagating gaps ("once a gap, always
       a gap"). Deterministic, and every row de-gaps back to its input (re-checked).
    """
    n = len(seqs)
    # score matrix (symmetric); pick center by total score
    score = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = float(aligner.align(seqs[i], seqs[j])[0].score)
            score[i][j] = score[j][i] = s
    center = max(range(n), key=lambda i: (sum(score[i]), -i))

    # msa_rows[0] is always the (progressively gapped) center row.
    order = [center] + [i for i in range(n) if i != center]
    msa_rows = [seqs[center]]
    row_index = [center]  # which original sequence each msa row corresponds to
    for i in order[1:]:
        best = aligner.align(seqs[center], seqs[i])[0]
        c_aln, s_aln = str(best[0]), str(best[1])
        msa_rows = _merge(msa_rows, c_aln, s_aln)
        row_index.append(i)
    return center, order, msa_rows, row_index


def _merge(msa_rows: list[str], c_aln: str, s_aln: str) -> list[str]:
    """Merge a fresh (center vs new-seq) pairwise alignment into an existing MSA.

    ``msa_rows[0]`` is the current gapped center; ``c_aln`` is the center as gapped in the
    new pairwise alignment and ``s_aln`` the new sequence. Reconcile the two gap patterns
    of the shared center, emitting a column at a time.
    """
    n_existing = len(msa_rows)
    center_row = msa_rows[0]
    out = [[] for _ in range(n_existing + 1)]
    i = j = 0
    while i < len(center_row) or j < len(c_aln):
        ci = center_row[i] if i < len(center_row) else None
        cj = c_aln[j] if j < len(c_aln) else None
        if ci == "-":
            # Existing MSA already has an insertion column here (from a prior sequence):
            # keep all existing rows, the new sequence gets a gap.
            for r in range(n_existing):
                out[r].append(msa_rows[r][i])
            out[n_existing].append("-")
            i += 1
        elif cj == "-":
            # The new pairwise introduces an insertion relative to the center: existing
            # rows get a gap, the new sequence contributes its residue.
            for r in range(n_existing):
                out[r].append("-")
            out[n_existing].append(s_aln[j])
            j += 1
        else:
            # Both point at a real (shared) center residue -> one aligned column.
            for r in range(n_existing):
                out[r].append(msa_rows[r][i])
            out[n_existing].append(s_aln[j])
            i += 1
            j += 1
    return ["".join(r) for r in out]


def sequence_align(
    sequences=None,
    seq_a=None,
    seq_b=None,
    mode: str = "global",
    seq_type: str = "auto",
) -> ToolResult:
    # ---- assemble the sequence list ----
    if sequences is None:
        if seq_a is not None and seq_b is not None:
            sequences = [seq_a, seq_b]
        else:
            return _validation_error(
                "Provide 'sequences' (a list of >= 2) or both 'seq_a' and 'seq_b'."
            )
    if not isinstance(sequences, (list, tuple)):
        return _validation_error("'sequences' must be a list of sequence strings.")
    if len(sequences) < 2:
        return _validation_error("Alignment needs at least 2 sequences.")
    if len(sequences) > _MAX_SEQUENCES:
        return _validation_error(
            f"Too many sequences ({len(sequences)} > {_MAX_SEQUENCES})."
        )

    seqs = [_clean(s) for s in sequences]
    for i, s in enumerate(seqs):
        if not s:
            return _validation_error(f"sequence[{i}] is empty after cleaning whitespace.")
        if len(s) > _MAX_SEQ_LEN:
            return _validation_error(
                f"sequence[{i}] length {len(s)} exceeds the {_MAX_SEQ_LEN} cap."
            )

    mode = str(mode).strip().lower()
    if mode not in ("global", "local"):
        return _validation_error("'mode' must be 'global' or 'local'.")

    seq_type = str(seq_type).strip().lower()
    if seq_type == "auto":
        detected = _detect_type(seqs)
        if detected is None:
            return _validation_error(
                "Could not auto-detect DNA/protein; characters fit neither alphabet. "
                "Pass seq_type explicitly."
            )
        seq_type = detected
    elif seq_type not in ("dna", "protein"):
        return _validation_error("'seq_type' must be 'dna', 'protein', or 'auto'.")

    err = _validate_alphabet(seqs, seq_type)
    if err:
        return err

    try:
        aligner, scoring = _build_aligner(mode, seq_type)
        algorithm = ("Needleman-Wunsch (global)" if mode == "global"
                     else "Smith-Waterman (local)")

        # ---------------------------------------------------------- pairwise
        if len(seqs) == 2:
            score, row_a, row_b, identities, aln_len = _pairwise(aligner, seqs[0], seqs[1])
            pct = round(100.0 * identities / aln_len, 4) if aln_len else 0.0
            return ToolResult(success=True, output={
                "kind": "pairwise",
                "mode": mode,
                "seq_type": seq_type,
                "algorithm": algorithm,
                "scoring": scoring,
                "score": score,
                "alignment": [row_a, row_b],
                "identities": identities,
                "alignment_length": aln_len,
                "percent_identity": pct,
            })

        # ---------------------------------------------------------- multiple
        # Center-star always merges via global pairwise alignments (local would leave
        # unaligned flanks that cannot be merged into a consistent MSA).
        if mode == "local":
            aligner.mode = "global"
        center, order, msa_rows, row_index = _center_star(aligner, seqs)

        # HONESTY: independently re-verify every row de-gaps to its original input.
        consistent = all(
            msa_rows[r].replace("-", "") == seqs[row_index[r]]
            for r in range(len(msa_rows))
        )
        if not consistent:
            return _execution_error(
                "center-star merge produced an inconsistent MSA (a row did not de-gap to "
                "its input); refusing to emit a fabricated alignment."
            )
        aln_len = len(msa_rows[0])
        # percent identity = fully-conserved columns / columns.
        conserved = 0
        for col in range(aln_len):
            column = {msa_rows[r][col] for r in range(len(msa_rows))}
            if len(column) == 1 and "-" not in column:
                conserved += 1
        pct = round(100.0 * conserved / aln_len, 4) if aln_len else 0.0

        # reorder rows back to input order for a stable, caller-friendly output
        by_input = [None] * len(seqs)
        for r, idx in enumerate(row_index):
            by_input[idx] = msa_rows[r]
        return ToolResult(success=True, output={
            "kind": "multiple",
            "mode": "global",
            "seq_type": seq_type,
            "algorithm": "center-star progressive (global pairwise merges)",
            "scoring": scoring,
            "center_index": center,
            "alignment": by_input,
            "conserved_columns": conserved,
            "alignment_length": aln_len,
            "percent_identity": pct,
            "consistent": consistent,
        })

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"sequence_align failed: {exc}")
