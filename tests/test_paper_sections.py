import asyncio

from propab.paper_compiler import collect_figure_object_ids, compile_references_latex
from propab.paper_sections import generate_prose_sections, render_paper_tex


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


def test_generate_prose_sections_no_llm_honest_abstract_when_zero_confirmed() -> None:
    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id="00000000-0000-0000-0000-000000000001",
            question="Compare optimizers A and B on synthetic losses.",
            prior={"key_papers": []},
            synthesis={
                "ledger": {"confirmed": [], "refuted": [], "inconclusive": ["h1", "h2"]},
            },
        )

        assert "confirmed=0" in out["abstract"]
        assert "inconclusive=2" in out["abstract"]
        assert "No hypothesis met" in out["abstract"]

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
