"""Evolve — the loop (WS-1): `Engine.step()` / `Engine.run()` over islands.

    parents  = island.sample()                    # favour winners, redirect off crowded families
    program  = mutator.mutate(parents, problem)   # LLM edits GENERATOR CODE (never raises)
    verdict  = self.evaluate(program)             # sandboxed run + cheap exact verify (engine.py)
    island.insert(program)                        # bounded top-k, deduped
    if problem.is_improvement(verdict): ledger.record(...)

WHY MIGRATION IS LATE AND STINGY
--------------------------------
The obvious design — every island sees the global champion, often — is the one that kills the search.
It converges every lineage onto the first approach that looked good, which is how you end up with
eight islands all polishing the same attractive-but-incomplete idea. So:

  * A mutation prompt is built ONLY from its own island's parents. The global best is never broadcast
    into a prompt. Islands must develop independently, and independence is the point of having them.
  * No migration at all during a warm-up phase (`migrate_warmup_steps`, default 5 migration
    intervals): cross-pollinating before an island has an idea of its own just copies the leader.
  * After warm-up, a migration event moves ONE champion into ONE neighbour (a ring), not a broadcast.
    Ideas diffuse slowly, and most islands are still working on their own thing at any moment.
  * `reset_weakest` reseeds from the problem's seeds plus a RANDOM surviving island's champion — not
    the global best (`reseed_from_global_best=True` restores the greedier behaviour if wanted). A
    fresh island that starts from the current favourite is not a fresh lineage.

ROBUSTNESS IS THE FEATURE
-------------------------
Mutated code crashes, hangs, allocates forever, and returns garbage; the model returns prose; the
sandbox times out. Over ten thousand steps all of these WILL happen, and none is an error path — they
are the ordinary cost of sampling program space. The only thing that must never happen is the loop
dying. Every seam (mutate, run, verify, is_improvement, record) is therefore guarded, and a failure
simply scores -inf and gets evicted.

The one thing that is NOT swallowed is a lost result: if the ledger cannot accept a verified
improvement, we log the whole record and keep it in `pending_records`. A discovery must never vanish
into an exception handler.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import random
from typing import Any

from .engine import Engine, EngineConfig, Mutator
from .island import (
    DEFAULT_CAPACITY,
    DEFAULT_FAMILY_BIAS,
    DEFAULT_REDISCOVERY_PENALTY,
    DEFAULT_TEMPERATURE,
    NEG_INF,
    REDISCOVERY_KEY,
    FamilyRegistry,
    ProgramIsland,
    clone,
    score_of,
)
from .ledger import Ledger, Record
from .mutator import NOOP_CODE
from .problem import INVALID, Problem, Verdict
from .program import ExecResult, Program, ProgramRunner

logger = logging.getLogger(__name__)

#: Migration intervals to wait before any cross-pollination. Islands need to become *different* from
#: each other before trading is worth anything.
DEFAULT_MIGRATE_WARMUP_INTERVALS = 5

DEFAULT_PARENTS_PER_STEP = 2


class EvolutionEngine(Engine):
    """The FunSearch-class loop over `ProgramIsland`s. Fills in `Engine.step`/`Engine.run`."""

    def __init__(
        self,
        problem: Problem,
        mutator: Mutator,
        runner: ProgramRunner,
        ledger: Ledger,
        config: EngineConfig | None = None,
        *,
        rng: random.Random | None = None,
        island_capacity: int = DEFAULT_CAPACITY,
        temperature: float = DEFAULT_TEMPERATURE,
        family_bias: float = DEFAULT_FAMILY_BIAS,
        rediscovery_penalty: float = DEFAULT_REDISCOVERY_PENALTY,
        parents_per_step: int = DEFAULT_PARENTS_PER_STEP,
        migrate_warmup_steps: int | None = None,
        reseed_from_global_best: bool = False,
        auditor: Any = None,
    ) -> None:
        super().__init__(problem, mutator, runner, ledger, config)
        # The adversarial layer. If it is None, every Record we build carries `audit=None`, and the
        # real Ledger refuses it ("no_audit") — i.e. the default is to bank NOTHING. That is
        # deliberate: an auditor nobody is forced to call is decoration.
        self.auditor = auditor
        self.audit_kills = 0
        self._rng = rng or random.Random()
        self.islands: list[ProgramIsland] = [
            ProgramIsland(
                i,
                capacity=island_capacity,
                temperature=temperature,
                family_bias=family_bias,
                rediscovery_penalty=rediscovery_penalty,
                rng=self._rng,
            )
            for i in range(max(1, self.config.islands))
        ]
        self.families = FamilyRegistry()
        self.parents_per_step = max(1, parents_per_step)
        self.reseed_from_global_best = reseed_from_global_best
        self.migrate_warmup_steps = (
            migrate_warmup_steps
            if migrate_warmup_steps is not None
            else DEFAULT_MIGRATE_WARMUP_INTERVALS * max(1, self.config.migrate_every)
        )

        self.steps = 0
        self.improvements = 0
        self.migrations = 0
        self.resets = 0
        self.record_failures = 0
        #: Verified improvements the ledger refused or failed to take. Never silently dropped.
        self.pending_records: list[Record] = []

        self._seed_pool: list[Program] = []
        self._seed_output_sigs: set[str] = set()
        self._seeded = False
        self._stopped = False
        self._verifier_fp: str | None = None

    # ---------------------------------------------------------------- seeding

    def seed(self) -> None:
        """Evaluate the problem's seed programs once, then stock every island with a private copy.

        Seeds are evaluated (not just inserted) so the population starts with real verifier scores —
        an unscored seed would be indistinguishable from junk at sampling time. Their outputs are
        fingerprinted so we can later tell when a "discovery" is really just the seed again.
        """
        if self._seeded:
            return
        self._seeded = True

        for code in self._seed_codes():
            program = Program(code=code, generation=0)
            verdict, result = self._evaluate_guarded(program)
            self._apply_verdict(program, verdict)
            candidate = self._winning_candidate(result, verdict) if verdict.valid else None
            sig = _output_signature(candidate)
            if sig is not None:
                self._seed_output_sigs.add(sig)
            self._seed_pool.append(program)
            self.families.observe(program)
            # A seed that already beats best_known means best_known is wrong, not that we discovered
            # something — but that is `is_improvement`'s call to make, not ours. Run the check.
            self._maybe_record(program, verdict, result, candidate)

        for island in self.islands:
            for program in self._seed_pool:
                island.insert(clone(program))

    def _seed_codes(self) -> list[str]:
        try:
            codes = self.problem.seed_programs() or []
        except Exception:  # noqa: BLE001
            logger.exception("evolve: problem.seed_programs() raised; starting from an empty pool")
            return []
        return [c for c in codes if isinstance(c, str) and c.strip()]

    # ---------------------------------------------------------------- the loop

    def step(self) -> Verdict:
        """One mutation: sample -> mutate -> evaluate -> insert -> maybe record. Never raises."""
        if not self._seeded:
            self.seed()

        island = self.islands[self.steps % len(self.islands)]
        self.steps += 1

        parents = self._sample_guarded(island)
        program = self._mutate_guarded(parents)
        verdict, result = self._evaluate_guarded(program)
        self._apply_verdict(program, verdict)

        candidate = self._winning_candidate(result, verdict) if verdict.valid else None
        sig = _output_signature(candidate)
        if sig is not None and sig in self._seed_output_sigs:
            # Scores well, discovers nothing: it re-derived a seed. Kept (it may still be a useful
            # scaffold) but penalised as a parent so its family cannot take over the island.
            program.detail[REDISCOVERY_KEY] = True

        island.insert(program)
        self.families.observe(program)
        self._maybe_record(program, verdict, result, candidate)
        return verdict

    def run(self) -> None:
        """Step until `config.max_steps` (None = until `request_stop()`), migrating and resetting."""
        self._stopped = False
        cfg = self.config
        while not self._stopped:
            if cfg.max_steps is not None and self.steps >= cfg.max_steps:
                break
            self.step()
            if cfg.migrate_every > 0 and self.steps % cfg.migrate_every == 0:
                self.migrate()
            if cfg.reset_weakest_every > 0 and self.steps % cfg.reset_weakest_every == 0:
                self.reset_weakest()

    def request_stop(self) -> None:
        """Halt `run()` at the next step boundary. `max_steps=None` means 'run until stopped'."""
        self._stopped = True

    # ---------------------------------------------------------------- diversity

    def migrate(self) -> int:
        """Copy ONE island's champion into its ring neighbour. Returns the number of programs moved.

        Deliberately minimal (see the module docstring): late, one champion at a time, never a
        broadcast of the global best. Copies rather than moves — migration must not weaken the source
        — and the destination dedupes by `Program.id`, so a repeat migration is a no-op.
        """
        if len(self.islands) < 2 or self.steps < self.migrate_warmup_steps:
            return 0
        order = list(range(len(self.islands)))
        self._rng.shuffle(order)
        for src in order:
            champion = self.islands[src].best()
            if champion is None or not math.isfinite(score_of(champion)):
                continue  # an island with nothing verified has nothing worth spreading
            dest = self.islands[(src + 1) % len(self.islands)]
            if dest.contains(champion.id):
                continue
            dest.insert(clone(champion))
            self.migrations += 1
            return 1
        return 0

    def reset_weakest(self) -> int | None:
        """Wipe the worst island and restock it. Returns the island index, or None if skipped.

        An island that has converged on nothing is dead budget: every sample comes from the same
        failed lineage. Resetting buys a fresh one. It is restocked from the problem's seeds plus one
        champion — by default from a RANDOM surviving island, not the global best, so a reset does
        not just clone the current favourite into yet another island.
        """
        if len(self.islands) < 2:
            return None  # resetting the only island would throw the whole population away
        weakest = min(self.islands, key=lambda isl: (isl.best_score(), isl.index))
        survivors = [isl for isl in self.islands if isl is not weakest]

        donor: Program | None = None
        if self.reseed_from_global_best:
            donor = self.best_program()
        elif survivors:
            donor = self._rng.choice(survivors).best()

        restock = [clone(p) for p in self._seed_pool]
        if donor is not None and math.isfinite(score_of(donor)):
            restock.append(clone(donor))
        weakest.reset(restock)
        self.resets += 1
        return weakest.index

    # ---------------------------------------------------------------- guards

    def _sample_guarded(self, island: ProgramIsland) -> list[Program]:
        try:
            return island.sample(self.parents_per_step)
        except Exception:  # noqa: BLE001
            logger.exception("evolve: island.sample failed; mutating from scratch")
            return []

    def _mutate_guarded(self, parents: list[Program]) -> Program:
        """The Mutator protocol says mutate() never raises. Trust, but do not bet the run on it."""
        hint = self.families.exploration_hint()
        set_hint = getattr(self.mutator, "set_exploration_hint", None)
        if callable(set_hint):
            try:
                set_hint(hint)
            except Exception:  # noqa: BLE001
                logger.exception("evolve: set_exploration_hint failed")
        try:
            program = self.mutator.mutate(parents, self.problem)
        except Exception as exc:  # noqa: BLE001
            logger.exception("evolve: mutator raised (it must not); using a no-op program")
            program = _noop_program(parents, f"mutator_raised: {type(exc).__name__}: {exc}")
        if not isinstance(program, Program):
            logger.error("evolve: mutator returned %r, not a Program", type(program).__name__)
            program = _noop_program(parents, "mutator_returned_non_program")
        return program

    def _evaluate_guarded(self, program: Program) -> tuple[Verdict, ExecResult]:
        """`Engine.evaluate` runs the sandbox and the verifier. A hostile program, a broken runner, or
        a verifier that raises on garbage must all cost one step, not the run."""
        try:
            return self.evaluate(program)
        except Exception as exc:  # noqa: BLE001
            logger.debug("evolve: evaluate failed: %s: %s", type(exc).__name__, exc)
            return INVALID, ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _apply_verdict(program: Program, verdict: Verdict) -> None:
        program.score = verdict.score
        program.valid = verdict.valid

    def _winning_candidate(self, result: ExecResult, verdict: Verdict) -> object | None:
        """Re-derive WHICH emitted candidate earned the verdict.

        `Engine.evaluate` (frozen) hands back only the best Verdict and the raw ExecResult — not the
        candidate that produced it — but a Record without the candidate is not independently
        re-checkable, and re-checkability is the entire credibility model. `verify` is cheap,
        deterministic and total by contract, so re-running it to find the match is safe.
        """
        if not verdict.valid:
            return None
        for candidate in result.candidates:
            try:
                v = self.problem.verify(candidate)
            except Exception:  # noqa: BLE001 — verify "never raises"; a target might disagree
                continue
            if v.valid and v.score == verdict.score:
                return candidate
        return None

    # ---------------------------------------------------------------- recording

    def _maybe_record(
        self,
        program: Program,
        verdict: Verdict,
        result: ExecResult,
        candidate: object | None,
    ) -> None:
        if not verdict.valid:
            return
        try:
            if not self.problem.is_improvement(verdict):
                return
        except Exception:  # noqa: BLE001
            logger.exception("evolve: problem.is_improvement raised; treating as 'not an improvement'")
            return

        record = self._build_record(program, verdict, candidate)

        # Adversarial audit BEFORE anything is banked. A kill here is a normal, expected event —
        # most "discoveries" a search produces are artifacts, and this is the layer that says so.
        record.audit = self._run_audit(program, verdict, candidate)
        if not record.audit_passed:
            self.audit_kills += 1
            logger.warning(
                "evolve: AUDIT KILLED a claimed improvement from %s (score=%s). kills=%s",
                program.id,
                verdict.score,
                (record.audit or {}).get("kills"),
            )
            return

        try:
            accepted = self.ledger.record(record)
        except Exception:  # noqa: BLE001
            # A VERIFIED improvement the ledger could not take is the one thing we refuse to lose:
            # log it in full (recoverable from the log alone) and hold it in memory.
            self.record_failures += 1
            self.pending_records.append(record)
            logger.error(
                "evolve: ledger.record FAILED on a verified improvement; retained in "
                "pending_records. Record follows:\n%s",
                _safe_json(record),
                exc_info=True,
            )
            return
        if accepted:
            self.improvements += 1
        else:
            logger.info("evolve: ledger rejected %s as a duplicate/rediscovery", program.id)

    def _run_audit(
        self, program: Program, verdict: Verdict, candidate: object | None
    ) -> dict[str, Any] | None:
        """Run the adversarial layer. Never raises.

        Returns the AuditReport as a dict, or None when no auditor is wired (in which case the real
        Ledger refuses the Record). A CRASHING auditor is a KILL, never a pass: if the thing whose
        job is to catch our mistakes is itself broken, we have learned nothing about the claim.
        """
        if self.auditor is None:
            return None
        try:
            report = self.auditor.audit(self.problem, verdict, candidate, program)
        except Exception as exc:  # noqa: BLE001 — a broken auditor certifies nothing
            logger.exception("evolve: auditor.audit raised; treating as a KILL")
            return {"passed": False, "kills": [f"auditor_crashed: {type(exc).__name__}: {exc}"]}
        to_json = getattr(report, "to_json", None)
        if callable(to_json):
            try:
                raw = to_json()
                return json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:  # noqa: BLE001
                logger.exception("evolve: AuditReport.to_json() failed; treating as a KILL")
                return {"passed": False, "kills": ["audit_report_unserializable"]}
        return {"passed": bool(getattr(report, "passed", False)),
                "kills": list(getattr(report, "kills", []) or [])}

    def _build_record(
        self, program: Program, verdict: Verdict, candidate: object | None
    ) -> Record:
        try:
            best_known = float(self.problem.best_known())
        except Exception:  # noqa: BLE001
            logger.exception("evolve: problem.best_known() raised; recording it as -inf")
            best_known = NEG_INF
        notes = "" if candidate is not None else "candidate could not be re-derived from ExecResult"
        return Record(
            problem=getattr(self.problem, "name", type(self.problem).__name__),
            score=verdict.score,
            best_known_at_time=best_known,
            candidate=candidate,
            witness=dict(verdict.detail),
            program_code=program.code,
            program_id=program.id,
            verifier_fingerprint=self.verifier_fingerprint(),
            generation=program.generation,
            notes=notes,
        )

    def verifier_fingerprint(self) -> str:
        """Hash of the verifier's source — pins WHAT checked a result, so a third party can tell
        whether the checker changed under them."""
        if self._verifier_fp is None:
            try:
                source = inspect.getsource(type(self.problem).verify)
            except (OSError, TypeError):  # pragma: no cover — e.g. a C-implemented verify
                self._verifier_fp = "unknown"
            else:
                self._verifier_fp = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return self._verifier_fp

    # ---------------------------------------------------------------- inspection

    def best_program(self) -> Program | None:
        best: Program | None = None
        for island in self.islands:
            champion = island.best()
            if champion is None:
                continue
            if best is None or score_of(champion) > score_of(best):
                best = champion
        return best

    def best_score(self) -> float:
        best = self.best_program()
        return NEG_INF if best is None else score_of(best)

    def population(self) -> list[Program]:
        """Every program alive in any island, best first."""
        everyone: dict[str, Program] = {}
        for island in self.islands:
            for program in island.programs():
                current = everyone.get(program.id)
                if current is None or score_of(program) > score_of(current):
                    everyone[program.id] = program
        return sorted(everyone.values(), key=score_of, reverse=True)

    def stats(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "best_score": self.best_score(),
            "improvements": self.improvements,
            "migrations": self.migrations,
            "resets": self.resets,
            "record_failures": self.record_failures,
            "islands": [len(i) for i in self.islands],
            "island_best": [i.best_score() for i in self.islands],
            "families": self.families.counts(),
            "dominant_family": self.families.dominant(),
        }


def _noop_program(parents: list[Program], reason: str) -> Program:
    return Program(
        code=NOOP_CODE,
        generation=1 + max((p.generation for p in parents), default=0),
        island=parents[0].island if parents else 0,
        parents=[p.id for p in parents],
        detail={"mutator_failure": reason},
    )


def _output_signature(candidate: object) -> str | None:
    """A stable fingerprint of what a program actually emitted.

    Used only to notice that a program re-derived a seed's own output — an "improvement" that is
    really the construction we started from. `repr` is enough: it is deterministic in-process for the
    JSON-ish structures candidates are made of, and a hash collision here costs one sampling penalty,
    not a false result.
    """
    if candidate is None:
        return None
    try:
        blob = repr(candidate)
    except Exception:  # noqa: BLE001 — a candidate's __repr__ is program-supplied, hence hostile
        return None
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:16]


def _safe_json(record: Record) -> str:
    try:
        return record.to_json()
    except Exception as exc:  # noqa: BLE001 — never lose the record to a serialization bug
        return f"<unserializable Record: {type(exc).__name__}: {exc}; repr={record!r}>"
