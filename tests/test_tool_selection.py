from propab.tool_selection import score_spec_relevance, select_tool_and_params


def test_score_prefers_matching_tool() -> None:
    specs = [
        {"name": "numeric_summary", "description": "mean and variance", "params": {}},
        {"name": "train_model", "description": "Train a neural network with Adam", "params": {}},
    ]
    hyp = "We should fine-tune the neural network using Adam optimizer."
    assert score_spec_relevance(hyp, specs[1]) > score_spec_relevance(hyp, specs[0])


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
