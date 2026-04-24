from propab.tool_selection import select_tool_steps


def _fake_specs() -> list[dict]:
    return [
        {
            "name": "alpha_tool",
            "description": "first alpha computation",
            "params": {"x": {"required": False}},
            "example": {"params": {"x": 1}},
        },
        {
            "name": "beta_tool",
            "description": "beta second",
            "params": {"y": {"required": False}},
            "example": {"params": {"y": 2}},
        },
        {
            "name": "gamma_tool",
            "description": "gamma third",
            "params": {"z": {"required": False}},
            "example": {"params": {"z": 3}},
        },
    ]


def test_select_tool_steps_respects_exclude() -> None:
    specs = _fake_specs()
    hyp = {"rank": 1}
    first = select_tool_steps(
        specs,
        hypothesis_text="alpha beta gamma computation",
        hypothesis=hyp,
        max_tools=2,
    )
    names = {t for t, _ in first}
    second = select_tool_steps(
        specs,
        hypothesis_text="alpha beta gamma computation",
        hypothesis=hyp,
        max_tools=2,
        exclude_tool_names=names,
    )
    assert not (names & {t for t, _ in second})
