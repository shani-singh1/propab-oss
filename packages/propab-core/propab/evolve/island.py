"""Evolve — the population (WS-1): islands + the approach-family registry.

Islands, rather than one global population, are what keep program-space search *exploring*. A single
population collapses onto the first lineage that works: every parent sample comes from the same
family, every mutation is a variation on one idea, and the search quietly stops finding anything new.
N islands run N lineages in parallel; the engine migrates winners between them late and sparingly,
and resets the weakest (see engine_impl.py). Diversity is the asset — this module protects it.

Three mechanisms, in increasing order of how much they actually matter:

1. **Boltzmann selection** (softmax over the VERIFIER's score). High scorers are favoured, but nothing
   with a finite score is locked out, so a lineage that is merely *currently* behind can still breed.
   Logits are divided by the score spread, which makes `temperature` scale-free: an island scoring in
   the millions and one scoring in [0, 1] behave identically at T=1, instead of the former collapsing
   to pure argmax (exp of a huge gap saturates) and destroying the very diversity islands exist for.

2. **Approach families.** Programs are grouped by the mathematical IDEA they embody ("algebraic /
   cyclic", "concatenation", "shortening a known code", "random + local search", …) — not by
   superficial code similarity. Islands alone do not prevent every island from independently
   rediscovering the same idea. So sampling is biased *against* whichever family currently dominates:
   a family holding many of an island's survivors gets down-weighted per member, redirecting parent
   selection toward underexplored formulations. `family_bias` tunes it: 0 = off, 1 = every family
   drawn with equal total probability regardless of how many members it has.

3. **Anti-circularity.** A family that scores well merely by re-deriving a seed construction must not
   be allowed to take over the population — that is motion without discovery. Programs the engine
   flagged as reproducing a seed's own output are down-weighted as parents (`rediscovery_penalty`).
   NOTE: this is *sampling pressure*, not the correctness gate. Rejecting a trivial rediscovery as a
   RESULT is `Problem.is_improvement`'s job and stays there.

The score used everywhere here is the verifier's, and only the verifier's. A program's own claims
about itself — printouts, returned "scores", comments — are never read.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import Counter

from .program import Program

DEFAULT_CAPACITY = 32
DEFAULT_TEMPERATURE = 1.0
# Per-member down-weighting of crowded families. Total sampling mass of a family with n members
# scales as n**(1 - family_bias): 0 => no redirect, 1 => families drawn equally regardless of size.
DEFAULT_FAMILY_BIAS = 0.5
# Multiplier on a parent that merely reproduces a seed's output. Not zero: such a program may still
# be a useful scaffold to mutate away from — it just must not dominate.
DEFAULT_REDISCOVERY_PENALTY = 0.25

NEG_INF = float("-inf")

UNKNOWN_FAMILY = "unknown"
FAMILY_KEY = "family"
REDISCOVERY_KEY = "rediscovers_seed"


def score_of(program: Program) -> float:
    """The program's verified score, normalized so it is always orderable.

    A verifier — or a program returning `nan` — can poison every comparison in the island: `nan`
    compares False against everything, so sorting and max() silently do the wrong thing. Treat any
    non-number as "never scored": -inf.
    """
    score = program.score
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return NEG_INF
    score = float(score)
    if math.isnan(score):
        return NEG_INF
    return score


def family_of(program: Program) -> str:
    """The approach family the mutator tagged this program with (see mutator.normalize_family)."""
    family = program.detail.get(FAMILY_KEY)
    if isinstance(family, str) and family.strip():
        return family.strip()
    return UNKNOWN_FAMILY


def is_rediscovery(program: Program) -> bool:
    """True if the engine flagged this program as re-deriving a seed's own output."""
    return program.detail.get(REDISCOVERY_KEY) is True


def clone(program: Program) -> Program:
    """A deep-enough copy: fresh `parents`/`detail` containers.

    `dataclasses.replace` would share those mutable fields between copies, so mutating one island's
    program would corrupt another's. Islands must own their members outright.
    """
    return Program(
        code=program.code,
        score=program.score,
        valid=program.valid,
        generation=program.generation,
        island=program.island,
        parents=list(program.parents),
        detail=dict(program.detail),
    )


class FamilyRegistry:
    """Cross-island bookkeeping: which mathematical ideas are we actually exploring?

    Counts only programs the verifier accepted — an idea that never produced a valid candidate is not
    an approach the search is "using", it is noise. The engine reads `exploration_hint()` and passes
    it to the mutator, which is how "one family is eating the search, go try something else" reaches
    the LLM at all.
    """

    #: A family is "dominant" once it holds this share of all valid programs...
    DOMINANCE_SHARE = 0.5
    #: ...provided we have seen at least this many valid programs (below it, "dominance" is noise).
    MIN_OBSERVATIONS = 4

    def __init__(self) -> None:
        self._valid: Counter[str] = Counter()          # family -> valid programs seen
        self._rediscoveries: Counter[str] = Counter()  # family -> valid programs re-deriving a seed
        self._seen_ids: set[str] = set()               # dedupe: the same code is one observation

    def observe(self, program: Program) -> None:
        if program.id in self._seen_ids:
            return
        self._seen_ids.add(program.id)
        if not program.valid or not math.isfinite(score_of(program)):
            return
        family = family_of(program)
        self._valid[family] += 1
        if is_rediscovery(program):
            self._rediscoveries[family] += 1

    def counts(self) -> dict[str, int]:
        return dict(self._valid)

    def total(self) -> int:
        return sum(self._valid.values())

    def dominant(self) -> str | None:
        """The named family that has taken over the search, if any."""
        total = self.total()
        if total < self.MIN_OBSERVATIONS:
            return None
        for family, n in self._valid.most_common():
            if family == UNKNOWN_FAMILY:
                continue  # untagged programs are not an "approach"; they cannot be dominant
            return family if n / total >= self.DOMINANCE_SHARE else None
        return None

    def underexplored(self, limit: int = 3) -> list[str]:
        """Named families we have barely tried, rarest first."""
        named = [(f, n) for f, n in self._valid.items() if f != UNKNOWN_FAMILY]
        named.sort(key=lambda item: (item[1], item[0]))
        return [f for f, _ in named[:limit]]

    def circular(self) -> list[str]:
        """Families whose valid programs are mostly just re-deriving the seeds."""
        out = []
        for family, redis in self._rediscoveries.items():
            n = self._valid.get(family, 0)
            if n >= 2 and redis / n > 0.5:
                out.append(family)
        return sorted(out)

    def exploration_hint(self) -> str | None:
        """Prompt text steering the mutator off a crowded/circular approach. None when healthy."""
        dominant = self.dominant()
        circular = self.circular()
        if not dominant and not circular:
            return None
        lines: list[str] = []
        if dominant:
            share = self._valid[dominant] / max(1, self.total())
            lines.append(
                f"DIVERSITY WARNING: the '{dominant}' family already accounts for "
                f"{share:.0%} of every program that works. Do NOT submit another one."
            )
        for family in circular:
            lines.append(
                f"DIVERSITY WARNING: the '{family}' family keeps re-deriving the seed construction. "
                f"Re-deriving a known construction scores points but discovers nothing."
            )
        rare = [f for f in self.underexplored() if f != dominant]
        if rare:
            lines.append(f"Under-explored so far: {', '.join(rare)}.")
        lines.append(
            "Use a DIFFERENT mathematical idea from the ones above — a different formulation, not a "
            "reworded version of the same one."
        )
        return "\n".join(lines)


class ProgramIsland:
    """One sub-population: a bounded top-k pool with diversity-aware, score-weighted sampling.

    Implements the `Island` protocol (engine.py): sample / insert / best / __len__.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        capacity: int = DEFAULT_CAPACITY,
        temperature: float = DEFAULT_TEMPERATURE,
        family_bias: float = DEFAULT_FAMILY_BIAS,
        rediscovery_penalty: float = DEFAULT_REDISCOVERY_PENALTY,
        rng: random.Random | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= family_bias <= 1.0:
            raise ValueError(f"family_bias must be in [0, 1], got {family_bias}")
        if not 0.0 < rediscovery_penalty <= 1.0:
            raise ValueError(f"rediscovery_penalty must be in (0, 1], got {rediscovery_penalty}")
        self.index = index
        self.capacity = capacity
        self.temperature = temperature
        self.family_bias = family_bias
        self.rediscovery_penalty = rediscovery_penalty
        self._rng = rng or random.Random()
        # id -> Program, in insertion order (dict ordering is the tie-break for eviction).
        self._members: dict[str, Program] = {}

    # ---------------------------------------------------------------- population

    def insert(self, program: Program) -> None:
        """Add a program, deduping by `Program.id` (i.e. by code) and keeping only the top-k.

        Identical code is the same program however often the model re-derives it — and it will
        re-derive it constantly, especially the no-op. We keep the better-scoring evaluation of the
        two rather than the newer one, so a flaky or timed-out re-run cannot demote a known winner.
        """
        existing = self._members.get(program.id)
        if existing is not None:
            if score_of(program) > score_of(existing):
                program.island = self.index
                self._members[program.id] = program
            return
        program.island = self.index
        self._members[program.id] = program
        self._evict()

    def _evict(self) -> None:
        """Drop the worst until we fit. Ties break oldest-first, so an island saturated with equally
        worthless programs keeps rotating in fresh ones instead of freezing."""
        overflow = len(self._members) - self.capacity
        if overflow <= 0:
            return
        # sorted() is stable => among equal scores, earlier-inserted comes first and is evicted.
        for victim in sorted(self._members.values(), key=score_of)[:overflow]:
            del self._members[victim.id]

    def reset(self, seeds: list[Program] | None = None) -> None:
        """Wipe the island and restock it. Used by the engine to kill a converged lineage."""
        self._members.clear()
        for program in seeds or []:
            self.insert(program)

    # ---------------------------------------------------------------- selection

    def sample(self, k: int = 2) -> list[Program]:
        """Sample up to `k` DISTINCT parents, favouring high scorers but redirecting away from
        whichever approach family has taken the island over.

        Distinct, because handing the mutator the same program twice wastes the prompt — two
        different parents are what let it recombine. Returns fewer than `k` (possibly none) if the
        island is smaller; the caller must cope, and the mutator does.
        """
        if k <= 0 or not self._members:
            return []
        pool = list(self._members.values())
        viable = [p for p in pool if math.isfinite(score_of(p))]
        # Breeding from programs that never produced a valid candidate is how an island burns its
        # budget — do it only when there is nothing else.
        candidates = viable or pool
        weights = self._weights(candidates)

        chosen: list[Program] = []
        for _ in range(min(k, len(candidates))):
            i = self._pick(weights)
            chosen.append(candidates.pop(i))
            weights.pop(i)
        return chosen

    def _weights(self, programs: list[Program]) -> list[float]:
        """Boltzmann weights, shifted by the max (so exp never overflows) and scaled by the spread
        (so `temperature` means the same thing whatever the objective's units), then corrected for
        family crowding and seed-rediscovery."""
        scores = [score_of(p) for p in programs]
        finite = [s for s in scores if math.isfinite(s)]
        if not finite:
            base = [1.0] * len(programs)  # nothing verified yet: uniform, never all-zero
        else:
            top = max(finite)
            spread = statistics.pstdev(finite) if len(finite) > 1 else 0.0
            scale = self.temperature * (spread or 1.0)
            base = [math.exp((s - top) / scale) if math.isfinite(s) else 0.0 for s in scores]

        # The island only ever holds survivors (it is a top-k pool), so counting members per family
        # IS counting the high scorers per family.
        crowding = Counter(family_of(p) for p in programs)
        out: list[float] = []
        for program, weight in zip(programs, base):
            if self.family_bias > 0.0:
                n = crowding[family_of(program)]
                weight *= n ** (-self.family_bias)
            if is_rediscovery(program):
                weight *= self.rediscovery_penalty
            out.append(weight)
        return out

    def _pick(self, weights: list[float]) -> int:
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0.0:
            return self._rng.randrange(len(weights))  # degenerate: fall back to uniform
        target = self._rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= target:
                return i
        return len(weights) - 1  # float dust

    # ---------------------------------------------------------------- inspection

    def best(self) -> Program | None:
        if not self._members:
            return None
        return max(self._members.values(), key=score_of)

    def best_score(self) -> float:
        best = self.best()
        return NEG_INF if best is None else score_of(best)

    def programs(self) -> list[Program]:
        """Members, best first."""
        return sorted(self._members.values(), key=score_of, reverse=True)

    def families(self) -> dict[str, int]:
        """This island's composition by approach family."""
        return dict(Counter(family_of(p) for p in self._members.values()))

    def contains(self, program_id: str) -> bool:
        return program_id in self._members

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"ProgramIsland(index={self.index}, size={len(self)}/{self.capacity}, "
            f"best={self.best_score():.4g})"
        )
