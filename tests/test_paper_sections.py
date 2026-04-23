from propab.paper_sections import render_paper_tex


def test_render_paper_tex_includes_sections() -> None:
    tex = render_paper_tex(
        title="Test title",
        abstract="Abs.",
        introduction="Intro.",
        methods_tex="\\section{Methods}\nX",
        discussion="Disc.",
    )
    assert "\\begin{abstract}" in tex
    assert "Test title" in tex
    assert "Abs." in tex
    assert "\\section{Methods}" in tex
