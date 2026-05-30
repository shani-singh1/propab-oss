import asyncio

from propab.paper_compiler import collect_figure_object_ids, compile_references_latex
from propab.paper_sections import generate_prose_sections, outcome_counts, render_paper_tex


def test_render_paper_tex_includes_sections() -> None:
    tex = render_paper_tex(
        title="Test title",
        abstract="Abs.",
        introduction="Intro.",
        methods_tex="\\section{Methods}\nX",
        results_tex="\\section{Results}\nY",
        figures_tex="",
        discussion="Disc.",
        conclusion="Conc.",
        references_tex="\\section{References}\n\\begin{thebibliography}{1}\\bibitem{x}Z\\end{thebibliography}",
    )
    assert "\\begin{abstract}" in tex
    assert "Test title" in tex
    assert "Abs." in tex
    assert "\\section{Methods}" in tex
    assert "\\section{Results}" in tex
    assert "\\section{Conclusion}" in tex
    assert "Conc." in tex
    assert "\\begin{thebibliography}" in tex


def test_compile_references_empty_prior() -> None:
    tex = compile_references_latex({})
    assert "References" in tex


def test_outcome_counts_prefers_db_counts_block() -> None:
    syn = {
        "counts": {"confirmed": 5, "refuted": 2, "inconclusive": 1, "tested": 8},
        "ledger": {"confirmed": ["a"], "refuted": [], "inconclusive": []},
    }
    c = outcome_counts(syn)
    assert (c["confirmed"], c["refuted"], c["inconclusive"], c["tested"]) == (5, 2, 1, 8)


def test_abstract_is_prose_not_ledger_tuple_and_matches_counts() -> None:
    """The abstract must read as prose and report the authoritative counts in words."""

    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id="00000000-0000-0000-0000-000000000003",
            question="Short synthetic campaign question.",
            prior={"key_papers": []},
            synthesis={"counts": {"confirmed": 22, "refuted": 2, "inconclusive": 0, "tested": 24}},
        )
        abstract = out["abstract"]
        # No machine-log ledger tuples in publishable prose.
        assert "confirmed=" not in abstract
        # Authoritative numbers expressed in words.
        assert "24 hypotheses" in abstract
        assert "22 were supported" in abstract
        assert "2 were refuted" in abstract
        assert out["title"]

    asyncio.run(_run())


def test_abstract_honest_when_zero_confirmed() -> None:
    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id="00000000-0000-0000-0000-000000000001",
            question="Compare optimizers A and B on synthetic losses.",
            prior={"key_papers": []},
            synthesis={"counts": {"confirmed": 0, "refuted": 0, "inconclusive": 2, "tested": 2}},
        )
        abstract = out["abstract"]
        assert "confirmed=" not in abstract
        assert "none met the significance threshold" in abstract

    asyncio.run(_run())


def test_abstract_honest_when_no_result_beats_baseline() -> None:
    """Honesty guard: 'supported' must not imply the baseline was beaten."""

    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id="00000000-0000-0000-0000-000000000004",
            question="Find the best MLP under 50k params for MNIST.",
            prior={"key_papers": []},
            synthesis={
                "counts": {"confirmed": 10, "refuted": 1, "inconclusive": 1, "tested": 12},
                "baseline_metric": 0.796,
                "best_metric": 0.698,
                "improvement_pct_over_baseline": -12.33,
                "metric_name": "val_accuracy",
            },
        )
        abstract = out["abstract"]
        assert "No configuration exceeded" in abstract
        assert "0.796" in abstract  # baseline stated explicitly
        assert "within-experiment comparisons" in abstract

    asyncio.run(_run())


def test_abstract_reports_improvement_when_baseline_beaten() -> None:
    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id="00000000-0000-0000-0000-000000000005",
            question="Find the best MLP under 50k params for MNIST.",
            prior={"key_papers": []},
            synthesis={
                "counts": {"confirmed": 8, "refuted": 0, "inconclusive": 0, "tested": 8},
                "baseline_metric": 0.796,
                "best_metric": 0.910,
                "improvement_pct_over_baseline": 14.32,
                "metric_name": "val_accuracy",
            },
        )
        abstract = out["abstract"]
        assert "improving on" in abstract
        assert "0.910" in abstract

    asyncio.run(_run())


def test_collect_figure_object_ids_dedupes() -> None:
    ids = collect_figure_object_ids(
        {
            "experiment_results": [
                {"figures": ["sessions/a/1.png", "b.png", "sessions/a/1.png"]},
                {"figures": ["b.png"]},
            ]
        }
    )
    assert ids == ["sessions/a/1.png", "b.png"]
