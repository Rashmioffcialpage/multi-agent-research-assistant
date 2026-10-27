import pytest

from research_assistant.tools.calculator import CalculatorError, calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("2 * (3 + 4) / 7", 2.0),
        ("-3 + 4", 1),
        ("2 ** 10", 1024),
        ("10 % 3", 1),
    ],
)
def test_calculate_correct_results(expression, expected):
    assert calculate(expression) == expected


def test_rejects_division_by_zero():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_rejects_arbitrary_code_execution():
    with pytest.raises(CalculatorError):
        calculate('__import__("os").system("echo pwned")')


def test_rejects_name_lookups():
    with pytest.raises(CalculatorError):
        calculate("some_variable + 1")


def test_rejects_invalid_syntax():
    with pytest.raises(CalculatorError):
        calculate("2 +* 3")
