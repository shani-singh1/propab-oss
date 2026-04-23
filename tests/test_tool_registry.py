from propab.tools.registry import ToolRegistry


def test_registry_discovers_core_tools() -> None:
    reg = ToolRegistry()
    names = {s["name"] for s in reg.get_all_specs()}
    assert "json_extract" in names
    assert "vector_dot" in names
    assert "category_counts" in names
    assert "numeric_summary" in names
