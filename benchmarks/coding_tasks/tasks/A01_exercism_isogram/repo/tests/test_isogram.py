from isogram import is_isogram


def test_empty_string_is_isogram() -> None:
    assert is_isogram("") is True


def test_word_with_repeated_letter_is_not_isogram() -> None:
    assert is_isogram("eleven") is False


def test_case_is_ignored() -> None:
    assert is_isogram("Alphabet") is False


def test_spaces_and_hyphens_are_ignored() -> None:
    assert is_isogram("six-year-old") is True


def test_repeated_letter_across_separator_is_detected() -> None:
    assert is_isogram("thumbscrew-japingly") is True
