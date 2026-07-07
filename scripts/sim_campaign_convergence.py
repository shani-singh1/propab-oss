"""Campaign-search convergence simulation — old vs new frontier scoring.

The full campaign loop needs the whole stack (postgres/redis/celery/workers), so
convergence can't be measured live in a unit test. This is the analog of the
literature n=100 eval: a lightweight, deterministic simulation of the search
dynamics that isolates the ONE thing the frontier policy controls — whether
dispatches are spent DEEPENING confirmed findings (narrowing a real result) or
adding fresh inconclusive BREADTH.

Model per generation:
  1. pick the next frontier node by the scoring-under-test,
  2. assign a mock verdict (a narrowing child of a confirmed node is itself more
     likely to confirm — real refinements of a true finding tend to hold),
  3. a confirmed node spawns a narrowing child (scope_delta/boundary) + a breadth
     root; an inconclusive node spawns a breadth child.
Metric: HypothesisTree.confirmed_lineage_depth() — mean confirmed-ancestry depth.
Rises when the search converges (deepens findings), stays ~1 when it only adds
shallow roots (the months-long failure). Same RNG seed for both policies.

Run: python scripts/sim_campaign_convergence.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "propab-core"))

from propab.hypothesis_tree import HypothesisNode, HypothesisTree  # noqa: E402
from propab.research_quality import estimate_closure_probability  # noqa: E402


def _legacy_score(tree: HypothesisTree, node: HypothesisNode) -> float:
    """The pre-fix _information_gain_score (confirmed parent = flat 0.45)."""
    counts = tree.theme_counts()
    theme_id = node.primary_theme or node.theme_id or "general"
    theme_count = counts.get(theme_id, 1)
    max_count = max(counts.values()) if counts else 1
    saturation = theme_count / max(max_count, 1)
    penalty = tree._scoring_context.theme_saturation_penalty
    relevance = float(node.question_relevance_score if node.question_relevance_score is not None else 0.5)
    novelty = max(0.05, 1.0 - saturation * penalty * 3.0)
    parent = tree.nodes.get(node.parent_id) if node.parent_id else None
    if parent is None:
        uncertainty = 0.55
    elif parent.verdict == "inconclusive":
        uncertainty = 0.85
    elif parent.verdict == "refuted":
        uncertainty = 0.70
    elif parent.verdict == "confirmed":
        uncertainty = 0.45
    else:
        uncertainty = 0.60
    coverage_bonus = max(0.0, 1.0 - min(1.0, theme_count / 12.0))
    lineage = float(node.lineage_length or tree.lineage_length(node.id))
    lineage_bonus = min(0.15, lineage * 0.03)
    info_gain = (
        0.30 * relevance + 0.25 * novelty + 0.25 * uncertainty
        + 0.10 * coverage_bonus + 0.10 * min(1.0, lineage_bonus * 5)
    )
    return info_gain * estimate_closure_probability(node, parent=parent)


def _pending(tree: HypothesisTree) -> list[HypothesisNode]:
    return [tree.nodes[n] for n in tree.frontier if n in tree.nodes and tree.nodes[n].verdict == "pending"]


def _add(tree: HypothesisTree, node: HypothesisNode) -> None:
    tree.nodes[node.id] = node
    tree.frontier.append(node.id)


def _child(parent: HypothesisNode, *, narrowing: bool, theme: str) -> HypothesisNode:
    return HypothesisNode(
        id=str(uuid4()),
        text=f"{'narrow ' if narrowing else 'breadth '}{theme} of {parent.id[:6]}",
        parent_id=parent.id,
        depth=parent.depth + 1,
        verdict="pending",
        theme_id=theme,
        primary_theme=theme,
        question_relevance_score=0.6,
        expansion_type="boundary" if narrowing else "alternative",
        scope_delta={"narrowed": "region"} if narrowing else None,
        lineage_length=(parent.lineage_length or parent.depth + 1) + 1,
    )


def run(policy: str, *, generations: int = 120, seed: int = 0) -> float:
    rng = random.Random(seed)
    tree = HypothesisTree()
    tree.set_scoring_context("threshold crossing search", [])
    for i in range(3):  # seed roots
        _add(tree, HypothesisNode(
            id=str(uuid4()), text=f"seed {i}", parent_id=None, depth=0, verdict="pending",
            theme_id=f"t{i}", primary_theme=f"t{i}", question_relevance_score=0.6,
        ))
    for _ in range(generations):
        pend = _pending(tree)
        if not pend:
            break
        if policy == "new":
            pend.sort(key=tree._information_gain_score, reverse=True)
        else:
            pend.sort(key=lambda n: _legacy_score(tree, n), reverse=True)
        node = pend[0]
        tree.frontier.remove(node.id)
        # A narrowing child of a confirmed parent is more likely to hold (a real
        # refinement of a true finding); a breadth probe is a coin flip.
        parent = tree.nodes.get(node.parent_id) if node.parent_id else None
        narrowing = bool(node.scope_delta)
        p_confirm = 0.72 if (narrowing and parent and parent.verdict == "confirmed") else 0.45
        node.verdict = "confirmed" if rng.random() < p_confirm else "inconclusive"
        theme = node.primary_theme or "t"
        if node.verdict == "confirmed":
            _add(tree, _child(node, narrowing=True, theme=theme))       # deepen
            _add(tree, HypothesisNode(                                    # + breadth root
                id=str(uuid4()), text="breadth root", parent_id=None, depth=0, verdict="pending",
                theme_id=f"b{rng.randint(0, 999)}", primary_theme=f"b{rng.randint(0, 999)}",
                question_relevance_score=0.6,
            ))
        else:
            _add(tree, _child(node, narrowing=False, theme=theme))
    return tree.confirmed_lineage_depth()


def main() -> None:
    seeds = range(8)
    old = [run("old", seed=s) for s in seeds]
    new = [run("new", seed=s) for s in seeds]
    mo, mn = sum(old) / len(old), sum(new) / len(new)
    print(f"mean confirmed-lineage depth over {len(list(seeds))} seeds, 120 generations:")
    print(f"  old policy: {mo:.2f}")
    print(f"  new policy: {mn:.2f}")
    print(f"  improvement: {mn - mo:+.2f} ({(mn / mo - 1) * 100:+.0f}%)" if mo else f"  new: {mn:.2f}")


if __name__ == "__main__":
    main()
