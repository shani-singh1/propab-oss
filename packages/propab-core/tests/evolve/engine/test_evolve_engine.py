"""Engine: the loop. Does the score go up, and does anything short of a power cut stop it?"""
from __future__ import annotations

import random

from conftest import (
    SEED_CODE,
    BrokenLedger,
    CrashingAuditor,
    CrashingLLM,
    CrashingProgramLLM,
    ExplodingRunner,
    HillClimbLLM,
    InProcRunner,
    KillingAuditor,
    PassingAuditor,
    RecordingLedger,
    ScriptedLLM,
    SumProblem,
    TimeoutRunner,
)

from propab.evolve.engine import EngineConfig
from propab.evolve.engine_impl import EvolutionEngine
from propab.evolve.island import FAMILY_KEY, REDISCOVERY_KEY, NEG_INF
from propab.evolve.mutator import NOOP_CODE, LLMMutator
from propab.evolve.program import Program

_UNSET = object()


def make_engine(
    problem=None,
    llm=None,
    runner=None,
    ledger=None,
    *,
    config=None,
    seed=7,
    auditor=_UNSET,
    **kwargs,
) -> EvolutionEngine:
    # Default to a passing auditor: these tests are about the loop and the record path. The audit
    # GATE itself (no auditor => bank nothing) is asserted explicitly below.
    return EvolutionEngine(
        problem or SumProblem(),
        LLMMutator(llm or HillClimbLLM()),
        runner or InProcRunner(),
        ledger,
        config or EngineConfig(islands=1, max_steps=60),
        rng=random.Random(seed),
        auditor=PassingAuditor() if auditor is _UNSET else auditor,
        **kwargs,
    )


def prog(tag: str, score: float) -> Program:
    """Code (and therefore id) is a function of `tag` alone — never of the score."""
    return Program(
        code=f"# {tag}\ndef build():\n    return [(1,)]\n",
        score=score,
        valid=score > NEG_INF,
        detail={FAMILY_KEY: "planted"},
    )


def improving() -> EngineConfig:
    """Long enough for the hill-climber to cross the record (30) from the seed (4).

    Not 1 point per step: Boltzmann selection and the family bias deliberately spend some samples off
    the leading lineage, which is the cost of not collapsing onto one idea.
    """
    return EngineConfig(islands=1, max_steps=140)


# --------------------------------------------------------------------------- seeding


def test_seed_stocks_every_island_with_scored_programs(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=4))
    engine.seed()

    assert all(len(island) == 1 for island in engine.islands)
    for island in engine.islands:
        seeded = island.best()
        assert seeded.code == SEED_CODE
        assert seeded.score == 4.0  # (1,1,1,1) — evaluated, not merely inserted
        assert seeded.valid


def test_islands_own_their_copies(tmp_path):
    """A shared Program object across islands means one island's bookkeeping corrupts another's."""
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=3))
    engine.seed()
    copies = [island.best() for island in engine.islands]
    assert len({id(c) for c in copies}) == 3
    assert [c.island for c in copies] == [0, 1, 2]


def test_seed_survives_a_problem_with_no_seeds(tmp_path):
    class NoSeeds(SumProblem):
        def seed_programs(self) -> list[str]:
            return []

    engine = make_engine(NoSeeds(), ledger=RecordingLedger(tmp_path))
    engine.step()  # must not raise: the mutator just writes from scratch
    assert engine.steps == 1


def test_seed_survives_seed_programs_exploding(tmp_path):
    class BadSeeds(SumProblem):
        def seed_programs(self) -> list[str]:
            raise RuntimeError("seed table unavailable")

    engine = make_engine(BadSeeds(), ledger=RecordingLedger(tmp_path))
    engine.step()
    assert engine.steps == 1


# --------------------------------------------------------------------------- the loop works


def test_score_improves_over_steps(tmp_path):
    """The whole bet: a dumb edit operator + a cheap exact verifier climbs on its own."""
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=improving())
    engine.seed()
    start = engine.best_score()

    engine.run()

    assert engine.best_score() > start, "the population never improved on its seed"
    assert engine.best_score() > SumProblem.BEST_KNOWN, engine.stats()


def test_verified_improvements_reach_the_ledger(tmp_path):
    ledger = RecordingLedger(tmp_path)
    engine = make_engine(ledger=ledger, config=improving())

    engine.run()

    assert ledger.records, "beat the record but wrote nothing to the ledger"
    rec = ledger.records[-1]
    assert rec.problem == "fake-sum"
    assert rec.score > rec.best_known_at_time == SumProblem.BEST_KNOWN
    assert rec.witness["values"]                      # the evidence a third party re-checks
    assert sum(rec.witness["values"]) == rec.score    # ...and it matches the claim
    assert rec.candidate is not None                  # the construction itself
    assert "def build" in rec.program_code            # the generator that found it
    assert len(rec.verifier_fingerprint) == 16        # WHAT checked it
    assert engine.improvements == len(ledger.records)


def test_a_rejected_record_is_not_counted_as_an_improvement(tmp_path):
    """The ledger is the authority on novelty: if it says 'rediscovery', we did not discover."""
    ledger = RecordingLedger(tmp_path, accept=False)
    engine = make_engine(ledger=ledger, config=improving())
    engine.run()
    assert ledger.records
    assert engine.improvements == 0


# --------------------------------------------------------------------------- the audit gate
# An auditor nobody is forced to call is decoration. These three prove it is forced.


def test_with_no_auditor_nothing_is_banked(tmp_path):
    """Default to REJECT. The run still beats the record — and still banks nothing, because no
    adversarial layer cleared it."""
    ledger = RecordingLedger(tmp_path)
    engine = make_engine(ledger=ledger, config=improving(), auditor=None)

    engine.run()

    assert engine.best_score() > SumProblem.BEST_KNOWN, "precondition: it did beat the record"
    assert not ledger.records, "banked a result that no auditor ever cleared"
    assert engine.improvements == 0
    assert engine.audit_kills > 0


def test_a_killed_audit_blocks_the_record(tmp_path):
    ledger = RecordingLedger(tmp_path)
    engine = make_engine(ledger=ledger, config=improving(), auditor=KillingAuditor())

    engine.run()

    assert engine.best_score() > SumProblem.BEST_KNOWN
    assert not ledger.records
    assert engine.improvements == 0
    assert engine.audit_kills > 0


def test_a_crashing_auditor_is_a_kill_not_a_pass(tmp_path):
    """If the thing whose job is to catch our mistakes is itself broken, we have learned NOTHING
    about the claim. Silence from a broken checker must never read as approval."""
    ledger = RecordingLedger(tmp_path)
    engine = make_engine(ledger=ledger, config=improving(), auditor=CrashingAuditor())

    engine.run()

    assert not ledger.records
    assert engine.improvements == 0
    assert engine.audit_kills > 0


def test_a_passing_audit_is_attached_to_the_record(tmp_path):
    """Every published result must carry the audit that cleared it."""
    ledger = RecordingLedger(tmp_path)
    engine = make_engine(ledger=ledger, config=improving())

    engine.run()

    assert ledger.records
    rec = ledger.records[-1]
    assert rec.audit_passed
    assert rec.audit["passed"] is True


def test_a_verified_improvement_is_never_lost_to_a_broken_ledger(tmp_path):
    """Disk full / WS-5 not landed. The run continues, but the result must survive in memory."""
    config = improving()
    engine = make_engine(ledger=BrokenLedger(tmp_path), config=config)

    engine.run()

    assert engine.steps == config.max_steps      # the loop did not die
    assert engine.record_failures > 0
    assert engine.pending_records                # nothing vanished into an exception handler
    assert engine.pending_records[0].score > SumProblem.BEST_KNOWN


def test_verifier_fingerprint_pins_the_checker(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path))
    other = make_engine(ledger=RecordingLedger(tmp_path))

    class DifferentVerifier(SumProblem):
        def verify(self, candidate):  # noqa: ANN001
            return super().verify(candidate)

    changed = make_engine(DifferentVerifier(), ledger=RecordingLedger(tmp_path))

    assert engine.verifier_fingerprint() == other.verifier_fingerprint()
    assert engine.verifier_fingerprint() != changed.verifier_fingerprint()


# --------------------------------------------------------------------------- adversarial junk


def test_crashing_programs_do_not_kill_the_loop(tmp_path):
    engine = make_engine(llm=CrashingProgramLLM(), ledger=RecordingLedger(tmp_path))

    engine.run()

    assert engine.steps == 60
    assert engine.best_score() == 4.0  # only the seed ever verified; the run carried on regardless


def test_a_hanging_program_does_not_kill_the_loop(tmp_path):
    """Every program times out in the sandbox — the engine sees ok=False and keeps going."""
    engine = make_engine(runner=TimeoutRunner(), ledger=RecordingLedger(tmp_path))
    engine.run()
    assert engine.steps == 60
    assert engine.best_score() == NEG_INF  # not even the seed survived, and that is fine


def test_a_broken_runner_does_not_kill_the_loop(tmp_path):
    """The runner contract says 'never propagate'. A runner that breaks it costs a step, not the run."""
    engine = make_engine(runner=ExplodingRunner(), ledger=RecordingLedger(tmp_path))
    engine.run()
    assert engine.steps == 60


def test_a_dead_llm_does_not_kill_the_loop(tmp_path):
    engine = make_engine(llm=CrashingLLM(), ledger=RecordingLedger(tmp_path))
    engine.run()
    assert engine.steps == 60
    assert engine.mutator.failures == 60
    assert engine.best_score() == 4.0  # the seed still stands


def test_a_mutator_that_raises_does_not_kill_the_loop(tmp_path):
    """The Mutator protocol forbids raising. Do not bet a ten-hour run on a third party honouring it."""

    class RogueMutator:
        def mutate(self, parents, problem):
            raise RuntimeError("I was told never to do this")

    engine = EvolutionEngine(
        SumProblem(),
        RogueMutator(),
        InProcRunner(),
        RecordingLedger(tmp_path),
        EngineConfig(islands=1, max_steps=5),
    )
    engine.run()

    assert engine.steps == 5
    assert any(p.code == NOOP_CODE for p in engine.population())


def test_a_mutator_returning_nonsense_does_not_kill_the_loop(tmp_path):
    class NonsenseMutator:
        def mutate(self, parents, problem):
            return "not a program"

    engine = EvolutionEngine(
        SumProblem(),
        NonsenseMutator(),
        InProcRunner(),
        RecordingLedger(tmp_path),
        EngineConfig(islands=1, max_steps=3),
    )
    engine.run()
    assert engine.steps == 3


def test_a_verifier_that_raises_on_garbage_does_not_kill_the_loop(tmp_path):
    class FragileProblem(SumProblem):
        def verify(self, candidate):  # noqa: ANN001
            raise ValueError("I refuse to look at that")

    engine = make_engine(FragileProblem(), ledger=RecordingLedger(tmp_path))
    engine.run()
    assert engine.steps == 60


def test_is_improvement_raising_is_treated_as_no_improvement(tmp_path):
    class FragileProblem(SumProblem):
        def is_improvement(self, verdict):  # noqa: ANN001
            raise ValueError("cannot reach the record table")

    ledger = RecordingLedger(tmp_path)
    engine = make_engine(FragileProblem(), ledger=ledger)
    engine.run()

    assert engine.steps == 60
    assert ledger.records == []  # cannot confirm an improvement => do not claim one


# --------------------------------------------------------------------------- run control


def test_run_respects_max_steps(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=2, max_steps=5))
    engine.run()
    assert engine.steps == 5


def test_request_stop_halts_an_open_ended_run(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=1, max_steps=None))

    original = engine.step

    def step_then_stop():
        verdict = original()
        if engine.steps >= 3:
            engine.request_stop()
        return verdict

    engine.step = step_then_stop  # type: ignore[method-assign]
    engine.run()

    assert engine.steps == 3


def test_steps_round_robin_across_islands(tmp_path):
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=3, max_steps=3, migrate_every=0, reset_weakest_every=0),
    )
    engine.run()
    # Each island got exactly one mutation on top of its seed.
    assert [len(i) for i in engine.islands] == [2, 2, 2]


# --------------------------------------------------------------------------- diversity


def test_prompts_never_broadcast_another_islands_champion(tmp_path):
    """Independence is the point of islands: an island must not be told the global favourite, or every
    lineage converges on the same attractive-but-incomplete idea."""
    llm = HillClimbLLM()
    engine = make_engine(
        llm=llm,
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=2, max_steps=1),
    )
    engine.seed()
    champion = prog("SECRET_CHAMPION", 999.0)
    engine.islands[1].insert(champion)

    engine.step()  # step 0 => island 0

    assert llm.prompts, "no prompt was built"
    assert all("SECRET_CHAMPION" not in p for p in llm.prompts)


def test_no_migration_during_the_warmup(tmp_path):
    """Cross-pollinating before an island has an idea of its own just copies the leader."""
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=2, max_steps=10, migrate_every=1, reset_weakest_every=0),
        migrate_warmup_steps=1000,
    )
    engine.run()
    assert engine.migrations == 0


def test_migration_copies_one_champion_to_its_ring_neighbour(tmp_path):
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=2, migrate_every=1),
        migrate_warmup_steps=0,
    )
    champion = prog("champ", 10.0)
    engine.islands[0].insert(champion)   # island 1 is empty => island 0 is the only viable source
    engine.steps = 50

    moved = engine.migrate()

    assert moved == 1
    assert engine.islands[1].contains(champion.id)
    assert engine.islands[0].contains(champion.id), "migration must copy, never weaken the source"
    assert engine.migrations == 1


def test_migration_is_idempotent(tmp_path):
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=2),
        migrate_warmup_steps=0,
    )
    engine.islands[0].insert(prog("champ", 10.0))
    engine.steps = 50

    engine.migrate()
    assert engine.migrate() == 0  # nothing new to send; dedupe makes a repeat a no-op


def test_migration_does_not_spread_junk(tmp_path):
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=2),
        migrate_warmup_steps=0,
    )
    engine.islands[0].insert(prog("junk", NEG_INF))
    engine.steps = 50
    assert engine.migrate() == 0


def test_reset_weakest_wipes_and_reseeds_the_dead_island(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=3))
    engine.seed()
    engine.islands[0].insert(prog("strong", 30.0))
    engine.islands[1].insert(prog("ok", 10.0))
    for i in range(5):
        engine.islands[2].insert(prog(f"junk{i}", NEG_INF))  # converged on nothing

    index = engine.reset_weakest()

    assert index == 2
    survivors = {p.code for p in engine.islands[2].programs()}
    assert not any(c.startswith("# junk") for c in survivors), "the dead lineage was not cleared"
    assert SEED_CODE in survivors, "the island was not reseeded from the problem's seeds"
    assert engine.resets == 1


def test_reset_does_not_simply_clone_the_global_favourite(tmp_path):
    """A fresh island restocked with the current leader is not a fresh lineage. The donor is a random
    surviving island's champion, so across resets it is NOT always the global best."""
    got_global_best = 0
    trials = 30
    for seed in range(trials):
        engine = make_engine(
            ledger=RecordingLedger(tmp_path),
            config=EngineConfig(islands=3),
            seed=seed,
        )
        engine.seed()
        best = prog("global_best", 100.0)
        engine.islands[0].insert(best)
        engine.islands[1].insert(prog("modest", 10.0))
        for i in range(3):
            engine.islands[2].insert(prog(f"junk{i}", NEG_INF))

        assert engine.reset_weakest() == 2
        if engine.islands[2].contains(best.id):
            got_global_best += 1

    assert got_global_best < trials, (
        "every reset cloned the global best — islands never get an independent lineage"
    )


def test_reset_from_global_best_when_explicitly_asked(tmp_path):
    engine = make_engine(
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=3),
        reseed_from_global_best=True,
    )
    engine.seed()
    best = prog("global_best", 100.0)
    engine.islands[0].insert(best)
    engine.islands[1].insert(prog("modest", 10.0))
    for i in range(3):
        engine.islands[2].insert(prog(f"junk{i}", NEG_INF))

    engine.reset_weakest()

    assert engine.islands[2].contains(best.id)


def test_reset_is_skipped_when_there_is_only_one_island(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=1))
    engine.seed()
    assert engine.reset_weakest() is None
    assert len(engine.islands[0]) == 1  # resetting the only island would throw everything away


def test_a_program_that_re_derives_a_seed_is_flagged_as_circular(tmp_path):
    """Different code, same output as the seed: scores fine, discovers nothing. It must be marked so
    its family cannot take over selection.

    Two *distinct* programs, because one program re-derived twice is still one program — the registry
    dedupes by id, and a single instance is a coincidence, not a circular family.
    """
    circular = ScriptedLLM(
        "```python\n# family: circular\ndef build():\n    x = 1\n    return [(x, x, x, x)]\n```",
        "```python\n# family: circular\ndef build():\n    y = 1\n    return [(y, y, y, y)]\n```",
    )
    engine = make_engine(
        llm=circular,
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=1, max_steps=2),
    )

    engine.run()

    rediscoveries = [p for p in engine.population() if p.detail.get(REDISCOVERY_KEY)]
    assert len(rediscoveries) == 2
    # Same score as the seed, because it IS the seed's output — motion without discovery.
    assert all(p.score == 4.0 for p in rediscoveries)
    assert engine.families.circular() == ["circular"]


def test_a_dominant_family_produces_an_exploration_hint(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=1, max_steps=10))
    engine.run()

    # Every child came from the same "hill-climb" idea, so the registry should be shouting about it.
    assert engine.families.dominant() == "hill-climb"
    hint = engine.families.exploration_hint()
    assert "DIVERSITY WARNING" in hint
    assert "hill-climb" in hint


def test_the_exploration_hint_reaches_the_prompt(tmp_path):
    llm = HillClimbLLM()
    engine = make_engine(
        llm=llm,
        ledger=RecordingLedger(tmp_path),
        config=EngineConfig(islands=1, max_steps=12),
    )
    engine.run()

    assert engine.families.dominant() == "hill-climb"
    assert any("DIVERSITY WARNING" in p for p in llm.prompts), (
        "the registry spotted a takeover but never told the model"
    )


# --------------------------------------------------------------------------- introspection


def test_stats_reports_the_run(tmp_path):
    engine = make_engine(ledger=RecordingLedger(tmp_path), config=EngineConfig(islands=2, max_steps=8))
    engine.run()

    stats = engine.stats()
    assert stats["steps"] == 8
    assert stats["islands"] == [len(i) for i in engine.islands]
    assert stats["best_score"] == engine.best_score()
    assert "hill-climb" in stats["families"]
