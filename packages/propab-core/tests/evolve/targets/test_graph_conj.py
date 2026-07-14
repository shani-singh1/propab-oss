"""Tests for Target B — counterexample hunting on open graph conjectures.

The load-bearing tests here are not the plumbing ones. They are:

  * `TestTranscription` — every OPEN conjecture must have margin <= 0 on ALL 994 connected graphs on
    <= 7 vertices. All four are verified in the literature for n <= 10, so a positive margin on a
    small graph proves WE mis-transcribed the conjecture. This is the guard against the failure mode
    that actually matters: a "refutation" of something we wrote down wrong.
  * `TestPositiveControl` — a conjecture with a PUBLISHED counterexample. The verifier must detect
    it, and a dumb LLM-free search must be able to REDISCOVER it. If it cannot, the engine is
    broken, not the conjecture.
  * `TestNegativeControl` — a THEOREM. The search must never find a counterexample. If it does, our
    evaluator is broken.
  * `TestDegeneracies` — the specific fake counterexamples a naive implementation would report.

No network at test time.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from propab.evolve.problem import Problem, Verdict
from propab.evolve.targets import conjectures as conj
from propab.evolve.targets.conjectures import MARGIN_EPS, Status
from propab.evolve.targets.graph_conj import (
    GraphConjectureProblem,
    decode_graph,
    double_star_counterexample,
    known_counterexample,
    problems,
    recheck_witness,
)

# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------


def _connected_atlas() -> list[nx.Graph]:
    """All connected graphs on 3..7 vertices (994 of them). Ships with networkx — no network."""
    return [
        g
        for g in nx.graph_atlas_g()
        if g.number_of_nodes() >= 3 and nx.is_connected(g)
    ]


ATLAS = _connected_atlas()
LIVE_KEYS = [c.key for c in conj.live_conjectures()]


# ------------------------------------------------------------------------------------------------
# the library itself
# ------------------------------------------------------------------------------------------------
class TestRegistry:
    def test_atlas_is_the_expected_size(self):
        # Guards the transcription test below: if this list silently emptied, that test would pass
        # vacuously and we would lose our only defence against a mis-stated conjecture.
        assert len(ATLAS) == 994

    def test_every_conjecture_is_sourced_and_status_checked(self):
        for c in conj.REGISTRY.values():
            assert c.source.strip(), f"{c.key} has no citation"
            assert c.statement.strip(), f"{c.key} has no statement"
            assert c.status_note.strip(), f"{c.key} has no status note"
            # a real ISO date, so a stale status is visible
            assert len(c.status_checked) == 10 and c.status_checked[4] == "-"

    def test_only_open_conjectures_are_live(self):
        for c in conj.REGISTRY.values():
            assert c.live == (c.status is Status.OPEN)

    def test_we_ship_live_targets_and_both_kinds_of_control(self):
        assert LIVE_KEYS, "no live hunting targets"
        controls = conj.controls()
        assert controls["positive"], "no positive control (a conjecture with a known counterexample)"
        assert controls["negative"], "no negative control (a theorem)"

    def test_no_unverified_conjecture_is_live(self):
        # The honesty rule: if we could not confirm a statement/status, it must never be hunted.
        for c in conj.REGISTRY.values():
            if c.status is Status.UNVERIFIED:
                assert not c.live

    def test_brouwer_is_marked_proven_not_open(self):
        # The cautionary entry. It was open for ~18 years, was still being treated as open in
        # January 2026, and was proven in mid-2026. Hunting it would be burning compute on a
        # counterexample that provably does not exist.
        assert conj.get("brouwer_laplacian").status is Status.PROVEN
        assert not conj.get("brouwer_laplacian").live


# ------------------------------------------------------------------------------------------------
# THE transcription guard
# ------------------------------------------------------------------------------------------------
class TestTranscription:
    """Every conjecture here is verified in the literature for n <= 10 (the theorems, for all n). So
    on every connected graph up to 7 vertices the margin MUST be <= 0. A violation means the bug is
    ours."""

    @pytest.mark.parametrize("key", LIVE_KEYS)
    def test_open_conjecture_never_fires_on_small_graphs(self, key):
        c = conj.get(key)
        worst, worst_g = -math.inf, None
        for g in ATLAS:
            if c.graph_class.violation(g) is not None:
                continue
            margin = c.margin(g)
            if margin > worst:
                worst, worst_g = margin, g
        assert worst <= MARGIN_EPS, (
            f"{key} reports a counterexample on {worst_g.number_of_nodes()} vertices "
            f"(edges={sorted(worst_g.edges())}, margin={worst:+.6g}). This conjecture is verified in "
            f"the literature for n <= {c.verified_up_to_n}, so OUR TRANSCRIPTION IS WRONG."
        )

    @pytest.mark.parametrize("key", ["hong_spectral_radius", "brouwer_laplacian"])
    def test_theorems_never_fire_on_small_graphs(self, key):
        c = conj.get(key)
        for g in ATLAS:
            if c.graph_class.violation(g) is None:
                assert c.margin(g) <= MARGIN_EPS, f"{key} is a THEOREM but fired — evaluator is broken"

    @pytest.mark.parametrize("key", LIVE_KEYS)
    def test_open_conjectures_are_tight_somewhere(self, key):
        """Each of these bounds is sharp. If our margin never gets near 0 on any small graph, we have
        probably transcribed a *different* (weaker) inequality that happens to be true."""
        c = conj.get(key)
        best = max(
            (c.margin(g) for g in ATLAS if c.graph_class.violation(g) is None),
            default=-math.inf,
        )
        assert best == pytest.approx(0.0, abs=1e-6), (
            f"{key} never attains equality on any graph up to 7 vertices (best margin {best:+.6g}); "
            f"the published bound is sharp, so this suggests a mis-transcription."
        )


# ------------------------------------------------------------------------------------------------
# positive control — a KNOWN counterexample must be found
# ------------------------------------------------------------------------------------------------
POSITIVE_KEY = "aouchiche_hansen_lam1_matching"


def _positive_control() -> GraphConjectureProblem:
    return GraphConjectureProblem(POSITIVE_KEY, allow_non_live=True)


class TestPositiveControl:
    def test_known_counterexample_is_detected(self):
        p = _positive_control()
        v = p.verify(known_counterexample())
        assert v.valid
        assert v.score > MARGIN_EPS, "the published counterexample was not detected"
        assert v.score == pytest.approx(0.080363, abs=1e-5)
        assert p.is_improvement(v)

    def test_known_counterexample_has_the_published_structure(self):
        g = known_counterexample()
        assert g.number_of_nodes() == 19          # Wagner's counterexample is on 19 vertices
        assert nx.is_connected(g)
        assert len(nx.max_weight_matching(g, maxcardinality=True)) == 2      # mu = 2
        lam1 = max(nx.adjacency_spectrum(g).real)
        assert lam1 == pytest.approx(math.sqrt(10), abs=1e-9)               # lambda_1 = sqrt(10)
        # lambda_1 + mu < sqrt(n-1) + 1
        assert lam1 + 2 < math.sqrt(18) + 1

    def test_a_graph_satisfying_the_conjecture_has_margin_at_most_zero(self):
        p = _positive_control()
        for g in (nx.path_graph(19), nx.cycle_graph(19), nx.complete_graph(12)):
            v = p.verify(g)
            assert v.valid
            assert v.score <= MARGIN_EPS
            assert not p.is_improvement(v)

    def test_the_star_is_exactly_tight(self):
        """The conjecture claims the star is optimal: lambda_1 = sqrt(n-1), mu = 1, margin == 0.
        Any counterexample must beat it, so if the star does not land on 0 we have the wrong bound."""
        p = _positive_control()
        for n in (11, 19, 30):
            v = p.verify(nx.star_graph(n - 1))
            assert v.score == pytest.approx(0.0, abs=1e-9)
            assert not p.is_improvement(v), "an equality case must never count as a refutation"

    def test_it_is_never_reported_as_a_discovery(self):
        """It is a real counterexample (is_improvement) but it was published in 2021, so it is a
        REDISCOVERY. Only `is_discovery` may be banked, and it must refuse."""
        p = _positive_control()
        v = p.verify(known_counterexample())
        assert p.is_improvement(v)
        assert not p.is_discovery(v)

    def test_the_balanced_double_star_family_refutes_it_with_a_closed_form(self):
        """The SIMPLEST counterexample family, checked against exact closed forms.

        S(a,a) = two adjacent centres with `a` leaves each (n = 2a+2):
            mu = 2                      (the two centres form a vertex cover of size 2)
            lambda_1 = (1+sqrt(1+4a))/2 (root of lambda^2 - lambda - a = 0)
        margin is exactly 0 at a=12 (n=26, lambda_1 = 4 exactly) and positive for every a >= 13.

        Checking the numerics against an independent closed form is the real point: it catches an
        eigenvalue/matching bug that a self-consistent implementation would happily hide.
        """
        p = _positive_control()
        for a in range(8, 20):
            g = double_star_counterexample(a)
            n = 2 * a + 2
            assert g.number_of_nodes() == n
            assert nx.is_tree(g)

            lam1_closed = (1 + math.sqrt(1 + 4 * a)) / 2
            expected = math.sqrt(n - 1) + 1 - (lam1_closed + 2)

            v = p.verify(g)
            assert v.valid
            assert v.detail["invariants"]["matching_number"] == 2
            assert v.detail["invariants"]["lambda_1"] == pytest.approx(lam1_closed, abs=1e-9)
            assert v.score == pytest.approx(expected, abs=1e-9)

            refutes = a >= 13
            assert (v.score > MARGIN_EPS) is refutes
            assert p.is_improvement(v) is refutes
            assert not p.is_discovery(v)            # published — never bankable

        # exactly on the boundary at a = 12
        assert p.verify(double_star_counterexample(12)).score == pytest.approx(0.0, abs=1e-9)

    def test_the_seed_programs_alone_rediscover_a_counterexample(self):
        """THE control that validates the engine's premise.

        Wagner refuted this conjecture with reinforcement learning and NO LLM. Our seeds are
        parameterised families and the verifier scores every member, so the seed sweep alone — no
        mutation, no LLM, no search heuristic at all — must already surface a counterexample. If the
        machinery cannot rediscover a KNOWN counterexample, then it will not find an unknown one,
        and the ENGINE is broken rather than the conjecture. That distinction is the single most
        important signal this target produces.

        Nothing is planted: the seeds emit textbook families (paths, stars, double stars,
        caterpillars, ...) and only the VERIFIER knows which of them refutes anything.
        """
        p = _positive_control()

        winners = []
        for src in p.seed_programs():
            ns: dict = {}
            exec(compile(src, "<seed>", "exec"), ns)      # noqa: S102 - our own seed source
            out = ns["build"]()
            for cand in out if isinstance(out, list) else [out]:
                v = p.verify(cand)
                if v.valid and p.is_improvement(v):
                    winners.append(v)

        assert winners, (
            "the seed sweep failed to rediscover ANY counterexample to a conjecture that was "
            "refuted in 2021 — the search machinery is broken, not the conjecture"
        )
        best = max(winners, key=lambda v: v.score)
        assert best.score > MARGIN_EPS
        assert best.detail["refutes"] is True
        assert recheck_witness(best.detail).score == pytest.approx(best.score, abs=1e-9)
        assert not p.is_discovery(best)              # a rediscovery is never a discovery

    def test_at_fixed_n_the_star_is_a_deceptive_local_optimum(self):
        """The landscape fact that constrains what the ENGINE may be — measured, not assumed.

        At n = 19 the star sits at margin EXACTLY 0, and every graph reached from it by pulling
        leaves onto a second centre (the adjacent double stars) is strictly WORSE, by up to 0.89.
        Wagner's counterexample (+0.08) sits on the far side of that valley.

        Measured consequence: at fixed n = 19, greedy object-space search never escapes this ridge
        (0/20 random seeds), and neither plain cross-entropy nor simulated annealing cracked it
        inside a unit-test budget. What DOES work is sweeping the family parameters -- which is
        exactly what a PROGRAM generator does, and why this engine searches program space rather
        than object space. Keep this test: if it ever goes green-by-accident, the landscape
        assumption behind the engine's design has changed.
        """
        p = _positive_control()

        def adjacent_double_star(a, b):
            g = nx.Graph([(0, 1)])
            g.add_edges_from((0, 2 + i) for i in range(a))
            g.add_edges_from((1, 2 + a + i) for i in range(b))
            return g

        star = p.verify(nx.star_graph(18)).score            # n = 19
        assert star == pytest.approx(0.0, abs=1e-9)

        valley = [p.verify(adjacent_double_star(17 - k, k)).score for k in range(1, 9)]
        assert all(s < star - 0.2 for s in valley), (
            "the n=19 double stars are supposed to form a VALLEY below the star; if they no longer "
            "do, the landscape behind the engine's search design has changed"
        )
        assert p.verify(known_counterexample()).score > star + 0.05


# ------------------------------------------------------------------------------------------------
# negative control — a theorem must never fire
# ------------------------------------------------------------------------------------------------
class TestNegativeControl:
    def test_hong_is_never_violated(self):
        """Hong's bound is a THEOREM. A positive margin would mean our evaluator or our candidate
        decoding is broken."""
        p = GraphConjectureProblem("hong_spectral_radius", allow_non_live=True)
        rng = random.Random(7)
        graphs = [nx.path_graph(15), nx.complete_graph(15), nx.star_graph(20), nx.petersen_graph()]
        graphs += [nx.random_labeled_tree(n, seed=rng.randrange(10**6)) for n in (11, 20, 30)]
        for _ in range(150):
            n = rng.randint(11, 30)
            g = nx.gnp_random_graph(n, rng.uniform(0.15, 0.9), seed=rng.randrange(10**6))
            if nx.is_connected(g):
                graphs.append(g)
        for g in graphs:
            v = p.verify(g)
            if v.valid:
                assert v.score <= MARGIN_EPS, f"Hong's THEOREM was 'refuted' — evaluator bug: {v.detail}"
                assert not p.is_improvement(v)


# ------------------------------------------------------------------------------------------------
# the degeneracies that would produce a FAKE counterexample
# ------------------------------------------------------------------------------------------------
class TestDegeneracies:
    def test_complete_graph_plus_isolated_vertex_does_not_refute_bollobas_nikiforov(self):
        """The trap. BN is stated "for G not a complete graph". K_n plus an isolated vertex is
        literally "not a complete graph", and lambda_1, lambda_2, m and omega are all unchanged by
        isolated vertices — so it inherits K_n's margin of +1 and a naive implementation "refutes" a
        2007 conjecture in one line. Requiring connectedness closes it."""
        p = GraphConjectureProblem("bollobas_nikiforov")
        g = nx.complete_graph(12)
        g.add_node(12)                                  # the isolated vertex
        v = p.verify(g)
        assert not v.valid, "K_n + isolated vertex must be rejected, not celebrated"
        assert "not connected" in v.detail["reason"]
        assert not p.is_improvement(v)

    def test_complete_graph_itself_is_out_of_class_for_bollobas_nikiforov(self):
        p = GraphConjectureProblem("bollobas_nikiforov")
        v = p.verify(nx.complete_graph(12))
        assert not v.valid
        assert "complete" in v.detail["reason"]

    def test_disconnected_graph_does_not_refute_efgw(self):
        """Two disjoint edges give s+ = 2 < 3 = n-1. EFGW is stated for CONNECTED graphs precisely
        because of this; an implementation that forgot the class check would 'refute' it instantly."""
        p = GraphConjectureProblem("efgw_min_square_energy")
        g = nx.Graph([(0, 1), (2, 3), (4, 5), (6, 7)])
        v = p.verify(g)
        assert not v.valid
        assert not p.is_improvement(v)

    def test_equality_cases_are_never_improvements(self):
        """EFGW is exactly tight on EVERY tree (bipartite => s+ = s- = m, and a tree has m = n-1).
        Floating-point noise must not turn an entire boundary family into 'refutations'."""
        p = GraphConjectureProblem("efgw_min_square_energy")
        rng = random.Random(3)
        for n in range(11, 41):
            for g in (nx.path_graph(n), nx.star_graph(n - 1),
                      nx.random_labeled_tree(n, seed=rng.randrange(10**6))):
                v = p.verify(g)
                assert v.valid
                assert v.score == pytest.approx(0.0, abs=1e-9), "trees must sit exactly on the boundary"
                assert not p.is_improvement(v), "an equality case is not a counterexample"


# ------------------------------------------------------------------------------------------------
# the Problem contract
# ------------------------------------------------------------------------------------------------
class TestProblemContract:
    def test_conforms_to_the_protocol(self):
        p = GraphConjectureProblem("elphick_linz_wocjan")
        assert isinstance(p, Problem)
        assert p.best_known() == 0.0
        assert p.name == "graph_conj:elphick_linz_wocjan"

    def test_refuses_to_hunt_a_non_live_conjecture(self):
        """The safety catch: you cannot accidentally point a campaign at a refuted or proven
        conjecture and 'discover' a counterexample."""
        for key in ("brouwer_laplacian", "hong_spectral_radius", POSITIVE_KEY):
            with pytest.raises(ValueError, match="not OPEN"):
                GraphConjectureProblem(key)
            GraphConjectureProblem(key, allow_non_live=True)   # explicit opt-in works

    def test_unknown_conjecture_raises_with_the_valid_keys(self):
        with pytest.raises(KeyError, match="unknown conjecture"):
            GraphConjectureProblem("no_such_conjecture")

    def test_describe_is_a_usable_prompt(self):
        for p in problems():
            d = p.describe()
            for required in ("margin", "COUNTEREXAMPLE", "build()", "CANDIDATE FORMAT"):
                assert required in d
            assert p.conjecture.statement in d
            assert p.conjecture.source in d
            assert p.conjecture.hunting_notes in d       # tell the LLM where NOT to look

    def test_problems_lists_only_live_targets_by_default(self):
        assert {p.conjecture.key for p in problems()} == set(LIVE_KEYS)
        assert all(p.conjecture.live for p in problems())
        assert len(problems(include_controls=True)) > len(problems())

    @pytest.mark.parametrize("key", LIVE_KEYS)
    def test_seed_programs_run_and_produce_valid_candidates(self, key):
        p = GraphConjectureProblem(key)
        seeds = p.seed_programs()
        assert seeds
        for src in seeds:
            ns: dict = {}
            exec(compile(src, "<seed>", "exec"), ns)      # noqa: S102 - our own seed source
            out = ns["build"]()
            cands = out if isinstance(out, list) else [out]
            assert cands, "a seed produced no candidates"
            verdicts = [p.verify(c) for c in cands]       # must never raise
            assert any(v.valid for v in verdicts), "a seed produced no in-class graph at all"
            # and no seed may accidentally hand us a counterexample to an OPEN conjecture: that would
            # mean either a mis-transcription or a hardcoded "answer"
            assert not any(p.is_improvement(v) for v in verdicts)

    def test_no_seed_hardcodes_wagners_specific_counterexample(self):
        """The seeds must be textbook FAMILIES, not a planted answer.

        Honest disclosure: the seed sweep DOES surface a counterexample to the positive control —
        the balanced double star on n >= 28 — and that is the point of
        `test_the_seed_programs_alone_rediscover_a_counterexample`. That is legitimate: the double
        star is a classic family any reasonable seed set contains, it is emitted as a parameter sweep
        with no knowledge of the conjecture, and only the VERIFIER identifies which member refutes.

        What would NOT be legitimate is shipping Wagner's specific 19-vertex graph as a seed and
        calling its rediscovery a search. This test pins that down.
        """
        p = _positive_control()
        target = nx.weisfeiler_lehman_graph_hash(known_counterexample())
        for src in p.seed_programs():
            ns: dict = {}
            exec(compile(src, "<seed>", "exec"), ns)      # noqa: S102
            out = ns["build"]()
            for c in out if isinstance(out, list) else [out]:
                g = decode_graph(c)
                if g.number_of_nodes() == 19:
                    assert nx.weisfeiler_lehman_graph_hash(g) != target


# ------------------------------------------------------------------------------------------------
# verify() must be total: a mutated program emits anything at all
# ------------------------------------------------------------------------------------------------
GARBAGE = [
    None,
    42,
    "not a graph",
    b"\x00\x01",
    [],
    {},
    {"nodes": 3},                                   # dict without "edges"
    [(0, 1), (1, "x")],                             # non-int vertex
    [(0, 0)],                                       # self-loop
    [(0, 1, 2)],                                    # not a pair
    [(-1, 3)],                                      # negative label
    [(0, 10**9)],                                   # absurd label
    [[0, 1], [0, 0]],                               # non-symmetric matrix-ish
    [[0, 2], [2, 0]],                               # non-binary
    float("nan"),
    nx.DiGraph([(0, 1)]),                           # wrong graph type (coerced, then class-checked)
    nx.Graph(),                                     # empty graph
    nx.complete_graph(500),                         # oversized: must be refused, not computed
    [(i, i + 1) for i in range(100000)],            # too many edges
    object(),
]


class TestGarbageSafety:
    @pytest.mark.parametrize("key", LIVE_KEYS)
    @pytest.mark.parametrize("junk", GARBAGE, ids=lambda x: type(x).__name__)
    def test_verify_never_raises_and_returns_invalid(self, key, junk):
        p = GraphConjectureProblem(key)
        v = p.verify(junk)                          # must not raise, ever
        assert isinstance(v, Verdict)
        assert not v.valid
        assert v.score == -math.inf
        assert "reason" in v.detail
        assert not p.is_improvement(v)

    def test_nan_in_adjacency_matrix_is_rejected(self):
        p = GraphConjectureProblem("efgw_min_square_energy")
        import numpy as np

        a = np.zeros((12, 12))
        a[0, 1] = a[1, 0] = float("nan")
        v = p.verify(a)
        assert not v.valid


# ------------------------------------------------------------------------------------------------
# the witness: a result must be re-checkable from the edge list alone
# ------------------------------------------------------------------------------------------------
class TestWitness:
    def test_witness_carries_the_object_and_both_sides(self):
        p = _positive_control()
        v = p.verify(known_counterexample())
        d = v.detail
        assert d["n"] == 19
        assert len(d["edges"]) == 18
        assert d["graph6"]                                   # canonical, pasteable into nauty/sage
        assert d["lhs"] - d["rhs"] == pytest.approx(d["margin"], abs=1e-12)
        assert d["refutes"] is True
        assert d["source"] and d["conjecture_statement"]

    def test_recheck_reproduces_the_margin_from_the_witness_alone(self):
        p = _positive_control()
        v = p.verify(known_counterexample())
        again = recheck_witness(v.detail)
        assert again.valid
        assert again.score == pytest.approx(v.score, abs=1e-12)

    def test_a_fabricated_margin_does_not_survive_the_recheck(self):
        """`is_improvement` recomputes from the witness and ignores the recorded arithmetic, so a
        verdict claiming a margin its graph does not have must be rejected."""
        p = GraphConjectureProblem("efgw_min_square_energy")
        v = p.verify(nx.path_graph(15))                      # a real graph, margin 0
        fake = Verdict(valid=True, score=99.0, detail={**v.detail, "margin": 99.0, "refutes": True})
        assert not p.is_improvement(fake)
        assert recheck_witness(fake.detail).score == pytest.approx(0.0, abs=1e-9)

    def test_a_witness_with_a_tampered_edge_list_is_rejected(self):
        p = _positive_control()
        v = p.verify(known_counterexample())
        tampered = {**v.detail, "edges": [[0, 1], [1, 2]]}    # a path on 3 of the 19 vertices
        assert recheck_witness(tampered).valid is False       # now disconnected => out of class
        assert not p.is_improvement(Verdict(valid=True, score=v.score, detail=tampered))

    def test_recheck_survives_a_corrupt_witness(self):
        for bad in ({}, {"conjecture": "nope", "n": 3, "edges": []},
                    {"conjecture": POSITIVE_KEY, "n": "x", "edges": None}):
            assert recheck_witness(bad).valid is False


# ------------------------------------------------------------------------------------------------
# the verifier must stay cheap — it is the thing the whole approach rests on
# ------------------------------------------------------------------------------------------------
class TestVerifierIsCheap:
    def test_verify_is_fast_even_on_an_adversarial_dense_graph(self):
        """The clique number is the only super-polynomial invariant. A Moon-Moser graph (complete
        multipartite, parts of size 3) has 3^(n/3) maximal cliques and would take ~minutes to
        enumerate; branch-and-bound plus the n cap must keep this in milliseconds."""
        import time

        p = GraphConjectureProblem("bollobas_nikiforov")
        worst = nx.complete_multipartite_graph(*([3] * 11))       # n = 33, just under the cap
        t0 = time.perf_counter()
        v = p.verify(worst)
        dt = time.perf_counter() - t0
        assert v.valid
        assert dt < 1.0, f"verifier took {dt:.2f}s on an adversarial graph — too slow to search with"

    def test_oversized_graphs_are_refused_by_the_class_not_computed(self):
        p = GraphConjectureProblem("bollobas_nikiforov")
        v = p.verify(nx.complete_multipartite_graph(*([3] * 20)))   # n = 60 > cap
        assert not v.valid
        assert "max_n" in v.detail["reason"]
