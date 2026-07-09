"""Tests for the B1 biology sequence/structure tool batch — correctness + honesty.

These tools were built to the domain-capabilities §0/§2a honesty spec: every reported
number (% identity, translation, GC, motif position, chain/residue counts, distances) is
computed from the real sequence/structure, never guessed; degenerate or invalid input
returns a ``validation_error`` and the tools never raise. The tests assert those
invariants alongside the happy path, driving the tools through the ``ToolRegistry`` so the
suite also confirms all four auto-register for the worker audience.
"""
from __future__ import annotations

from propab.tools.registry import ToolRegistry

_R = ToolRegistry()


def call(name, **params):
    return _R.call(name, params)


# ── registration ─────────────────────────────────────────────────────────────
def test_all_four_register_for_worker():
    names = {s["name"] for s in _R.get_for("worker")}
    assert {"sequence_align", "motif_scan", "translate_annotate",
            "structure_analysis"} <= names


# ── sequence_align ───────────────────────────────────────────────────────────
def test_align_near_identical_high_identity():
    r = call("sequence_align", seq_a="ACGTACGT", seq_b="ACGTTCGT",
             mode="global", seq_type="dna")
    assert r.success
    # 7 of 8 columns identical -> exactly 87.5%; a hand count reproduces it.
    assert r.output["identities"] == 7
    assert r.output["alignment_length"] == 8
    assert r.output["percent_identity"] == 87.5


def test_align_identity_matches_hand_count():
    # HONESTY: the tool's identity must equal an independent hand count on a tiny case.
    a, b = "ACGT", "ACGT"
    r = call("sequence_align", seq_a=a, seq_b=b, seq_type="dna")
    assert r.success
    row_a, row_b = r.output["alignment"]
    hand = sum(1 for x, y in zip(row_a, row_b) if x == y and x != "-")
    assert r.output["identities"] == hand == 4
    assert r.output["percent_identity"] == 100.0


def test_align_local_mode_and_score_real():
    r = call("sequence_align", seq_a="TTTTACGTACGTTTTT", seq_b="ACGTACGT",
             mode="local", seq_type="dna")
    assert r.success and r.output["mode"] == "local"
    # local alignment recovers the embedded exact match: all aligned columns identical.
    assert r.output["identities"] == r.output["alignment_length"] == 8


def test_align_multiple_is_consistent():
    r = call("sequence_align", sequences=["ACGTACGT", "ACGTTCGT", "ACGAACGT"])
    assert r.success and r.output["kind"] == "multiple"
    # HONESTY: every MSA row, gaps removed, equals its input sequence.
    assert r.output["consistent"] is True
    for row, original in zip(r.output["alignment"],
                             ["ACGTACGT", "ACGTTCGT", "ACGAACGT"]):
        assert row.replace("-", "") == original


def test_align_empty_sequence_is_validation_error():
    r = call("sequence_align", seq_a="", seq_b="ACGT")
    assert not r.success and r.error.type == "validation_error"


def test_align_invalid_alphabet_is_validation_error():
    r = call("sequence_align", seq_a="ACGTZZ", seq_b="ACGTAA", seq_type="dna")
    assert not r.success and r.error.type == "validation_error"


def test_align_single_sequence_is_validation_error():
    r = call("sequence_align", sequences=["ACGT"])
    assert not r.success and r.error.type == "validation_error"


# ── translate_annotate ───────────────────────────────────────────────────────
def test_translate_known_peptide():
    # ATGGCCATTGTAATGGGCCGCTGA -> M A I V M G R (to_stop drops the TGA stop).
    r = call("translate_annotate", op="translate",
             sequence="ATGGCCATTGTAATGGGCCGCTGA", to_stop=True)
    assert r.success and r.output["result"] == "MAIVMGR"


def test_translate_short_peptide():
    r = call("translate_annotate", op="translate", sequence="ATGGCC")
    assert r.success and r.output["result"] == "MA" and r.output["codons_translated"] == 2


def test_gc_content_known_sequence():
    r = call("translate_annotate", op="gc_content", sequence="ATGCATGC")
    assert r.success and r.output["result"] == 0.5 and r.output["gc_count"] == 4
    r2 = call("translate_annotate", op="gc_content", sequence="GGGG")
    assert r2.success and r2.output["result"] == 1.0


def test_reverse_complement_known():
    r = call("translate_annotate", op="reverse_complement", sequence="ATGC")
    assert r.success and r.output["result"] == "GCAT"


def test_codon_usage_counts():
    r = call("translate_annotate", op="codon_usage", sequence="ATGATGGCC")
    assert r.success and r.output["result"] == {"ATG": 2, "GCC": 1}


def test_translate_non_multiple_of_three_is_validation_error():
    # HONESTY: bad codon length is refused, not silently mis-translated.
    r = call("translate_annotate", op="translate", sequence="ATGGC")
    assert not r.success and r.error.type == "validation_error"


def test_translate_invalid_alphabet_is_validation_error():
    r = call("translate_annotate", op="translate", sequence="ATGZZZ")
    assert not r.success and r.error.type == "validation_error"


# ── motif_scan ───────────────────────────────────────────────────────────────
def test_motif_scan_finds_planted_motif_at_right_position():
    # Plant GAATTC starting at index 3; it must be found there and only there.
    seq = "AAA" + "GAATTC" + "TTT"
    r = call("motif_scan", op="consensus", sequence=seq, motif="GAATTC")
    assert r.success and r.output["n_matches"] == 1
    m = r.output["matches"][0]
    assert m["start"] == 3 and m["end"] == 9 and m["matched"] == "GAATTC"


def test_motif_scan_iupac_ambiguity():
    # TATAWAW (W = A/T) matches TATAAAT at index 0.
    r = call("motif_scan", op="consensus", sequence="TATAAATCCC", motif="TATAWAW")
    assert r.success and r.output["matches"][0]["start"] == 0


def test_motif_scan_orf_six_frames():
    # ATG GCC GGG TAA embedded at index 3 -> one +1-frame ORF translating to MAG.
    r = call("motif_scan", op="orf", sequence="AAAATGGCCGGGTAAAAA", min_aa=1)
    assert r.success and r.output["n_orfs"] >= 1
    orf = [o for o in r.output["orfs"] if o["protein"] == "MAG"]
    assert orf and orf[0]["start"] == 3 and orf[0]["strand"] == "+"


def test_motif_scan_restriction_sites():
    r = call("motif_scan", op="restriction", sequence="GAATTCGGATCC")
    hits = {(m["enzyme"], m["start"]) for m in r.output["matches"]}
    assert ("EcoRI", 0) in hits and ("BamHI", 6) in hits


def test_motif_scan_pwm_scores_are_real():
    # Identity PWM: score of a window = number of positions matching the diagonal.
    pwm = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    r = call("motif_scan", op="pwm", sequence="ACGT", pwm=pwm, threshold=4.0)
    assert r.success and r.output["matches"][0]["score"] == 4.0


def test_motif_scan_invalid_sequence_is_validation_error():
    r = call("motif_scan", op="consensus", sequence="ACGTQ", motif="ACG")
    assert not r.success and r.error.type == "validation_error"


def test_motif_scan_missing_motif_is_validation_error():
    r = call("motif_scan", op="consensus", sequence="ACGT")
    assert not r.success and r.error.type == "validation_error"


# ── structure_analysis ───────────────────────────────────────────────────────
# A tiny but valid PDB: chain A with 2 residues (ALA, GLY), chain B with 1 (SER),
# plus one HELIX annotation record.
_TINY_PDB = """HEADER    TEST STRUCTURE
HELIX    1   1 ALA A    1  GLY A    2  1                                   2
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.140   6.331  -5.169  1.00  0.00           C
ATOM      4  O   ALA A   1      13.665   7.031  -6.040  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.388   4.680  -4.588  1.00  0.00           C
ATOM      6  N   GLY A   2      13.844   5.750  -4.198  1.00  0.00           N
ATOM      7  CA  GLY A   2      15.286   5.914  -4.093  1.00  0.00           C
ATOM      8  C   GLY A   2      15.888   4.879  -3.156  1.00  0.00           C
ATOM      9  O   GLY A   2      15.181   4.262  -2.353  1.00  0.00           O
ATOM     10  N   SER B   1      17.191   4.697  -3.269  1.00  0.00           N
ATOM     11  CA  SER B   1      17.905   3.735  -2.444  1.00  0.00           C
ATOM     12  C   SER B   1      19.406   3.982  -2.532  1.00  0.00           C
ATOM     13  O   SER B   1      19.856   4.851  -3.278  1.00  0.00           O
ATOM     14  CB  SER B   1      17.595   2.317  -2.912  1.00  0.00           C
TER
END
"""


def test_structure_parses_chains_and_residue_counts():
    r = call("structure_analysis", structure=_TINY_PDB, format="pdb")
    assert r.success
    assert r.output["n_chains"] == 2
    assert r.output["total_residues"] == 3
    by_id = {c["id"]: c["residues"] for c in r.output["chains"]}
    assert by_id["A"] == 2 and by_id["B"] == 1


def test_structure_secondary_structure_from_records():
    # HONESTY: the one HELIX record is reported; nothing is invented.
    r = call("structure_analysis", structure=_TINY_PDB)
    ss = r.output["secondary_structure"]
    assert ss["available"] is True and ss["helices"] == 1 and ss["strands"] == 0


def test_structure_no_ss_records_reports_unavailable_not_fabricated():
    # Strip the HELIX record -> the tool must report SS unavailable, never guess.
    no_helix = "\n".join(l for l in _TINY_PDB.splitlines() if not l.startswith("HELIX"))
    r = call("structure_analysis", structure=no_helix, format="pdb")
    assert r.success and r.output["secondary_structure"]["available"] is False


def test_structure_pair_distance_is_computed_and_missing_is_flagged():
    r = call("structure_analysis", structure=_TINY_PDB,
             pairs=[["A", 1, "A", 2], ["A", 99, "B", 1]])
    pd = r.output["pair_distances"]
    # real CA-CA distance for the existing pair (~3.8 A), an error for the missing one.
    assert abs(pd[0]["distance"] - 3.799) < 0.01
    assert "error" in pd[1] and "distance" not in pd[1]


def test_structure_empty_input_is_validation_error():
    r = call("structure_analysis", structure="   ")
    assert not r.success and r.error.type == "validation_error"


def test_structure_unparseable_is_execution_error_not_guess():
    # HONESTY: garbage in -> a clear error, never a fabricated chain/residue count.
    r = call("structure_analysis", structure="not a structure at all", format="pdb")
    assert not r.success and r.error.type == "execution_error"
