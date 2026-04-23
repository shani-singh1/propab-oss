from propab.tools.mathematics.vector_dot import vector_dot


def test_vector_dot_basic() -> None:
    r = vector_dot([1.0, 2.0], [3.0, 4.0])
    assert r.success
    assert r.output["dot"] == 11.0
    assert r.output["length"] == 2


def test_vector_dot_mismatch() -> None:
    r = vector_dot([1.0], [1.0, 2.0])
    assert not r.success
