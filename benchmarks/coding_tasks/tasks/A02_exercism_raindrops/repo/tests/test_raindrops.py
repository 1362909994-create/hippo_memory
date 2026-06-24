from raindrops import convert


def test_factor_of_three() -> None:
    assert convert(3) == "Pling"


def test_factor_of_five() -> None:
    assert convert(10) == "Plang"


def test_factor_of_seven() -> None:
    assert convert(14) == "Plong"


def test_multiple_factors_are_concatenated() -> None:
    assert convert(105) == "PlingPlangPlong"


def test_no_factor_returns_digits() -> None:
    assert convert(34) == "34"
