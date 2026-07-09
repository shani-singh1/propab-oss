"""DNA translation / annotation primitives (pure, deterministic, honest).

B1 (domain-capabilities §2a): a ``translate_annotate`` tool over the everyday nucleotide
bookkeeping a biologist repeats across investigations — DNA->protein translation with the
correct codon table, reverse-complement / complement, GC content, codon usage and basic
composition stats — all via ``Bio.Seq`` / ``Bio.SeqUtils`` with exact counting.

Honesty by construction (domain-capabilities §0):
  * Translation uses the requested NCBI codon table (default 1, the standard code);
    the table id and whether translation stopped at a stop codon are reported.
  * GC content / codon usage / composition are exact counts over the real sequence —
    never estimated. The op + parameters used are always reported.
  * Codon length and alphabet are validated: a non-DNA character or a length that is not
    a multiple of 3 (for translate / codon_usage, unless allow_partial) returns a
    ``validation_error``. The tool never raises and never guesses on bad input.
  * Protein->DNA "back-translation" is intentionally NOT offered: it is ambiguous and any
    answer would be fabricated.
"""
from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

_DNA_ALPHABET = set("ACGTN")

_ALLOWED_OPS = frozenset(
    {"translate", "reverse_complement", "complement", "gc_content",
     "codon_usage", "composition"}
)

_MAX_SEQ_LEN = 5_000_000

TOOL_SPEC = {
    "name": "translate_annotate",
    "domain": "biology",
    "audience": "worker",
    "description": (
        "Deterministic DNA annotation via Bio.Seq. op is one of: translate (DNA->protein "
        "with a chosen NCBI codon table), reverse_complement, complement, gc_content, "
        "codon_usage (codon counts + frequencies over the coding frame), composition (base "
        "counts, GC/AT). Params: sequence (DNA over ACGTN), table (codon table id, default "
        "1), to_stop (translate: stop at the first stop codon), allow_partial (translate/"
        "codon_usage: tolerate a length not divisible by 3 by trimming the trailing partial "
        "codon). Validates alphabet + codon length; bad input -> validation_error."
    ),
    "params": {
        "op": {"type": "str", "required": True,
               "description": "translate | reverse_complement | complement | gc_content | "
                              "codon_usage | composition."},
        "sequence": {"type": "str", "required": True, "description": "DNA sequence over ACGTN."},
        "table": {"type": "int", "required": False,
                  "description": "NCBI codon table id (default 1)."},
        "to_stop": {"type": "bool", "required": False,
                    "description": "translate: stop at (and drop) the first stop codon."},
        "allow_partial": {"type": "bool", "required": False,
                          "description": "translate/codon_usage: trim a trailing partial codon."},
    },
    "output": {
        "op": "str — the operation performed",
        "result": "the primary result (op-specific: protein string, revcomp, fraction, dict)",
        "table": "int — codon table id (translate/codon_usage)",
        "to_stop": "bool — translate: whether it stopped at a stop codon",
        "codons_translated": "int — translate/codon_usage: number of whole codons used",
        "trailing_bases": "int — leftover bases not forming a whole codon",
        "length": "int — sequence length",
    },
    "example": {
        "params": {"op": "translate", "sequence": "ATGGCC"},
        "output": {"op": "translate", "result": "MA", "table": 1,
                   "codons_translated": 2, "trailing_bases": 0},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _clean(seq) -> str:
    return "".join(str(seq).split()).upper()


def _valid_table_id(table) -> bool:
    from Bio.Data import CodonTable

    return table in CodonTable.unambiguous_dna_by_id


def translate_annotate(
    op: str | None = None,
    sequence=None,
    table: int = 1,
    to_stop: bool = False,
    allow_partial: bool = False,
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
        from Bio.Seq import Seq

        bio = Seq(seq)

        # ---------------------------------------------------------- reverse_complement
        if op == "reverse_complement":
            return ToolResult(success=True, output={
                "op": op, "result": str(bio.reverse_complement()), "length": len(seq)})

        # ---------------------------------------------------------- complement
        if op == "complement":
            return ToolResult(success=True, output={
                "op": op, "result": str(bio.complement()), "length": len(seq)})

        # ---------------------------------------------------------- gc_content
        if op == "gc_content":
            from Bio.SeqUtils import gc_fraction

            gc = sum(seq.count(b) for b in "GC")
            at = sum(seq.count(b) for b in "AT")
            frac = float(gc_fraction(seq))  # ambiguous 'N' excluded by default
            return ToolResult(success=True, output={
                "op": op,
                "result": round(frac, 6),
                "gc_count": gc,
                "at_count": at,
                "length": len(seq),
                "method": "GC / (A+C+G+T); ambiguous bases excluded (Bio.SeqUtils.gc_fraction)",
            })

        # ---------------------------------------------------------- composition
        if op == "composition":
            counts = {b: seq.count(b) for b in "ACGTN"}
            gc = counts["G"] + counts["C"]
            at = counts["A"] + counts["T"]
            denom = gc + at
            return ToolResult(success=True, output={
                "op": op,
                "result": counts,
                "gc_count": gc,
                "at_count": at,
                "gc_fraction": round(gc / denom, 6) if denom else 0.0,
                "length": len(seq),
            })

        # ops below need a codon table -------------------------------------------------
        if not isinstance(table, int) or isinstance(table, bool) or not _valid_table_id(table):
            return _validation_error(
                f"'table' must be a valid NCBI codon table id (got {table!r})."
            )

        # ---------------------------------------------------------- translate
        if op == "translate":
            trailing = len(seq) % 3
            if trailing and not allow_partial:
                return _validation_error(
                    f"'sequence' length {len(seq)} is not a multiple of 3 "
                    f"({trailing} trailing base(s)); set allow_partial=True to trim it."
                )
            usable = seq[: len(seq) - trailing] if trailing else seq
            n_codons = len(usable) // 3
            protein = str(Seq(usable).translate(table=table, to_stop=to_stop))
            return ToolResult(success=True, output={
                "op": op,
                "result": protein,
                "table": table,
                "to_stop": bool(to_stop),
                "codons_translated": n_codons,
                "trailing_bases": trailing,
                "protein_length": len(protein),
                "length": len(seq),
            })

        # ---------------------------------------------------------- codon_usage
        if op == "codon_usage":
            trailing = len(seq) % 3
            if trailing and not allow_partial:
                return _validation_error(
                    f"'sequence' length {len(seq)} is not a multiple of 3 "
                    f"({trailing} trailing base(s)); set allow_partial=True to trim it."
                )
            usable = seq[: len(seq) - trailing] if trailing else seq
            codons = [usable[i:i + 3] for i in range(0, len(usable), 3)]
            counts: dict[str, int] = {}
            for c in codons:
                counts[c] = counts.get(c, 0) + 1
            total = len(codons)
            freqs = {c: round(n / total, 6) for c, n in counts.items()} if total else {}
            return ToolResult(success=True, output={
                "op": op,
                "result": counts,
                "frequencies": freqs,
                "table": table,
                "codons_translated": total,
                "trailing_bases": trailing,
                "length": len(seq),
            })

        return _validation_error(f"Unhandled op {op!r}.")

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"translate_annotate op '{op}' failed: {exc}")
