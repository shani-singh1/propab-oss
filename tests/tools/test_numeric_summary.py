from propab.tools.data_analysis.numeric_summary import numeric_summary


def test_numeric_summary_std() -> None:
    r = numeric_summary([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert r.success
    assert r.output["count"] == 8
    assert abs(r.output["mean"] - 5.0) < 1e-5
    assert r.output["min"] == 2.0
    assert r.output["max"] == 9.0
    assert r.output["std_sample"] > 0
