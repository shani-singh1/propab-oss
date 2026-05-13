"""LaTeX tabular rendering for tool params/outputs in campaign papers."""

from propab.config import Settings, _apply_profile
from propab.paper_compiler import (
    _tabular_payload_from_tool_output,
    latex_tabular_from_jsonish,
)


def test_latex_tabular_from_flat_dict() -> None:
    tex = latex_tabular_from_jsonish({"lr": 0.01, "epochs": 3, "note": "a&b"})
    assert tex
    assert "\\begin{tabular}" in tex
    assert r"\&" in tex or "a" in tex


def test_latex_tabular_from_list_of_dicts() -> None:
    rows = [{"cfg": "A", "acc": 0.9}, {"cfg": "B", "acc": 0.91}]
    tex = latex_tabular_from_jsonish(rows)
    assert tex
    assert "cfg" in tex
    assert "acc" in tex


def test_latex_tabular_from_numeric_list_short() -> None:
    tex = latex_tabular_from_jsonish([0.7, 0.7, 0.71])
    assert tex
    assert "\\textbf{Index}" in tex


def test_tabular_payload_prefers_inner_output() -> None:
    wrapped = {"success": True, "output": {"mean": 1.2, "std": 0.1}}
    inner = _tabular_payload_from_tool_output(wrapped)
    tab = latex_tabular_from_jsonish(inner)
    assert tab and "mean" in tab


def test_campaign_profile_caps_sandbox_retries_and_disables_timeout_rewrite() -> None:
    s = Settings(propab_profile="campaign")
    _apply_profile(s)
    assert s.sandbox_code_max_retries == 1
    assert s.sandbox_after_timeout_llm_rewrite is False
