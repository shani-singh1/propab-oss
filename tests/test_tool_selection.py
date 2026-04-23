from propab.tool_selection import score_spec_relevance, select_tool_and_params, select_tool_steps


def test_score_prefers_matching_tool() -> None:
    specs = [
        {"name": "numeric_summary", "description": "mean and variance", "params": {}},
        {"name": "train_model", "description": "Train a neural network with Adam", "params": {}},
    ]
    hyp = "We should fine-tune the neural network using Adam optimizer."
    assert score_spec_relevance(hyp, specs[1]) > score_spec_relevance(hyp, specs[0])


def test_select_tool_steps_returns_two_distinct_tools() -> None:
    specs = [
        {"name": "tool_a", "description": "alpha neural network", "params": {}, "example": {"params": {"u": 1}}},
        {"name": "tool_b", "description": "beta optimizer", "params": {}, "example": {"params": {"v": 2}}},
        {"name": "tool_c", "description": "gamma", "params": {}, "example": {}},
    ]
    out = select_tool_steps(
        specs,
        hypothesis_text="We study neural network training with an optimizer.",
        hypothesis={"rank": 1},
        max_tools=2,
    )
    assert len(out) == 2
    assert out[0][0] != out[1][0]
    assert out[0][1] != out[1][1]


def test_select_prefers_example_params() -> None:
    specs = [
        {
            "name": "z_first",
            "description": "neural network training",
            "params": {},
            "example": {},
        },
        {
            "name": "json_extract",
            "description": "parse json",
            "params": {},
            "example": {"params": {"data": {"x": 1}, "key": "x"}},
        },
    ]
    name, params = select_tool_and_params(specs, hypothesis_text="neural network adam", hypothesis={"rank": 1})
    assert name == "json_extract"
    assert params["key"] == "x"
