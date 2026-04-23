from propab.paper_compiler import collect_figure_object_ids, compile_references_latex
from propab.paper_sections import render_paper_tex


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
