from propab.sub_agent_plan import parse_llm_plan_steps


def _spec(name: str, required: bool) -> dict:
    return {
        "name": name,
        "description": f"tool {name}",
        "params": {"x": {"type": "int", "required": required}},
        "output": {},
        "example": {"params": {"x": 1}, "output": {}},
    }


def test_parse_llm_plan_steps_accepts_valid_json() -> None:
    specs = [_spec("alpha", True), _spec("beta", False)]
    raw = '{"steps":[{"type":"tool","tool":"alpha","params":{"x":3}},{"type":"tool","tool":"beta","params":{"x":2}}]}'
    out = parse_llm_plan_steps(raw, specs=specs, max_steps=4)
    assert out == [("alpha", {"x": 3}), ("beta", {"x": 2})]


def test_parse_llm_plan_steps_rejects_unknown_tool() -> None:
    specs = [_spec("alpha", True)]
    raw = '{"steps":[{"type":"tool","tool":"nope","params":{"x":1}}]}'
    assert parse_llm_plan_steps(raw, specs=specs, max_steps=4) is None


def test_parse_llm_plan_strips_markdown_fence() -> None:
    specs = [_spec("alpha", True)]
    raw = '```json\n{"steps":[{"type":"tool","tool":"alpha","params":{"x":1}}]}\n```'
    out = parse_llm_plan_steps(raw, specs=specs, max_steps=2)
    assert out == [("alpha", {"x": 1})]
