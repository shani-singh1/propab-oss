"""Campaign-convergence benchmark — drives the REAL convergence code offline.

Unlike a toy sim, this exercises the actual production paths that decide whether
a campaign converges: `apply_synthesis_to_frontier` (real parent resolution +
real duplicate filter + real frontier insertion), the real frontier selection
(`next_dispatch_candidate` / `_information_gain_score`), and real `update_node`.
The only mocks are the two things that genuinely need the live stack:
  - the LLM synthesis response — constructed here as a schema-valid parsed dict
    (a real orchestrator would get this from the LLM), proposing, for each
    confirmed frontier-eligible node, a scope-NARROWING child (explicit parent_id,
    boundary expansion, a strictly narrower numeric region);
  - the worker verdict — a ground-truth model where a genuine narrowing of a true
    finding tends to hold (p_confirm high) and a fresh breadth probe is a coin flip.

Headline metrics (all offline, deterministic per seed):
  - max_depth      : deepest confirmed lineage reached in the budget (convergence)
  - mean_depth     : mean confirmed-ancestry depth over confirmed nodes
  - narrow_reject  : fraction of proposed NARROWING children the dedup rejected
                     (should be ~0 — rejecting these is the convergence-killer)
  - children_ratio : children / (children + roots) among added nodes

Run: python scripts/bench_campaign_convergence.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "propab-core"))

from propab.belief_state import CampaignBeliefState  # noqa: E402
from propab.campaign_synthesis import apply_synthesis_to_frontier  # noqa: E402
from propab.hypothesis_tree import HypothesisTree  # noqa: E402

QUESTION = "Where does greedy F(n)/sqrt(n) first fall below 0.60 for n in [10000,50000]?"

_SCOPE = (
    "\nPopulation: greedy Sidon sequences\nDistribution: n in {lo}..{hi}\n"
    "Claimed generalization: crossing in [{lo},{hi}]\n"
    "Expected failure modes: nonmonotonic dip\nOOD test: dense n grid"
)


def _cand(slug: str, lo: int, hi: int, parent_id: str | None) -> dict:
    text = f"Greedy F(n)/sqrt(n) crosses 0.60 within n in [{lo},{hi}]." + _SCOPE.format(lo=lo, hi=hi)
    d = {
        "id": slug,
        "text": text,
        "test_methodology": "numeric_sweep over n",
        "expansion_type": "boundary",
        "why_follows_from_beliefs": "narrow the confirmed crossing region",
    }
    if parent_id:
        d["parent_id"] = parent_id
    return d


def _region(node) -> tuple[int, int]:
    import re
    m = re.search(r"\[(\d+),(\d+)\]", node.text)
    return (int(m.group(1)), int(m.group(2))) if m else (10000, 50000)


def run(*, generations: int = 40, seed: int = 0, node_cap: int = 90) -> dict:
    rng = random.Random(seed)
    tree = HypothesisTree()
    tree.set_scoring_context(QUESTION, [])
    state = CampaignBeliefState()
    # Seed one root covering the whole region.
    root = _cand("root", 10000, 50000, None)
    added = tree.add_seeds([root], generation=0)
    tree.add_to_frontier(added)

    narrow_proposed = 0
    narrow_rejected = 0
    child_ct = 0
    root_ct = 0
    counter = 0
    max_concurrent = 6

    for gen in range(1, generations + 1):
        # 1) DISPATCH WAVE: take the best up-to-max_concurrent pending frontier
        #    nodes (real frontier selection) and assign each a ground-truth verdict.
        wave = tree.next_batch(max_concurrent)
        if not wave:
            break
        for node in wave:
            lo, hi = _region(node)
            narrowing = (hi - lo) < 40000  # not the full-region root
            parent = tree.nodes.get(node.parent_id) if node.parent_id else None
            p = 0.75 if (narrowing and parent and parent.verdict == "confirmed") else 0.45
            verdict = "confirmed" if rng.random() < p else "inconclusive"
            tree.update_node(node.id, verdict, 0.8 if verdict == "confirmed" else 0.4,
                             'evidence={"verdict_reason": "sweep"};')

        # 2) SYNTHESIS: propose a strictly-narrower child for each confirmed,
        #    still-expandable node (the convergence move the LLM is told to make).
        #    Feed the REAL apply_synthesis_to_frontier so its real dedup + parent
        #    resolution + frontier insertion decide what survives.
        if len(tree.nodes) >= node_cap:  # real campaigns have a hypothesis cap;
            break                        # measure convergence within a fixed budget
        targets = [n for n in tree.nodes.values()
                   if n.verdict == "confirmed" and len(n.children) < 5 and n.depth < 8]
        cands = []
        for t in targets:
            lo, hi = _region(t)
            span = max(200, (hi - lo) // 3)
            c = lo + rng.randint(0, max(1, (hi - lo) - span))
            counter += 1
            cands.append(_cand(f"n{counter}", c, c + span, t.id))
        if not cands:
            if not tree.next_batch(1):
                break
            continue
        narrow_proposed += len(cands)
        parsed = {"beliefs": [], "frontier_candidates": cands, "direction_exhausted": False}
        # PROD relevance threshold (0.35), not 0.0 — the benchmark must exercise
        # the same gate production does, or it hides bugs (e.g. narrowing children
        # being relevance-dropped; investigation report §6d).
        _got, metrics = apply_synthesis_to_frontier(
            tree, state, parsed, question=QUESTION, generation=gen, relevance_threshold=0.35,
        )
        child_ct += int(metrics.get("n_added_as_children") or 0)
        root_ct += int(metrics.get("n_added_as_roots") or 0)
        narrow_rejected += int(metrics.get("n_rejected_duplicate") or 0)

    depths = [tree._confirmed_ancestry_depth(n) + 1
              for n in tree.nodes.values() if n.verdict == "confirmed"]
    return {
        "max_depth": max(depths) if depths else 0,
        "mean_depth": round(sum(depths) / len(depths), 2) if depths else 0.0,
        "narrow_reject": round(narrow_rejected / narrow_proposed, 3) if narrow_proposed else 0.0,
        "children_ratio": round(child_ct / (child_ct + root_ct), 3) if (child_ct + root_ct) else 0.0,
        "n_confirmed": len(depths),
    }


def main() -> None:
    seeds = range(10)
    rows = [run(seed=s) for s in seeds]
    def avg(k): return round(sum(r[k] for r in rows) / len(rows), 3)
    print(f"Campaign-convergence benchmark (real synthesis+dedup+frontier code), {len(rows)} seeds:")
    print(f"  max confirmed-lineage depth   : {avg('max_depth')}   (higher = converges deeper)")
    print(f"  mean confirmed-lineage depth  : {avg('mean_depth')}")
    print(f"  narrowing-dedup reject rate   : {avg('narrow_reject')}   (LOWER is better; ~0 ideal)")
    print(f"  children/(children+roots)     : {avg('children_ratio')}   (higher = deepening not breadth)")
    print(f"  confirmed nodes               : {avg('n_confirmed')}")


if __name__ == "__main__":
    main()
