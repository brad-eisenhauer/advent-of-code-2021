""" Advent of Code 2019, Day 04: https://adventofcode.com/2019/day/4 """

import pytest

from util import Timer


def main(timer: Timer):
    valid_range = range(271973, 785962)

    # Part 1, non-strict
    p1_result = sum(1 for n in valid_range if code_is_valid(n))
    print(p1_result)
    timer.check("Part 1")

    # Part 2, strict
    p2_result = sum(1 for n in valid_range if code_is_valid(n, strict=True))
    print(p2_result)
    timer.check("Part 2")


def code_is_valid(n: int, strict: bool = False) -> bool:
    # six digit number; no leading zeroes
    if n not in range(100000, 1000000):
        return False

    def generate_digits():
        temp = n
        while temp > 0:
            yield temp % 10
            temp //= 10

    # digits must be monotonically increasing
    digits = generate_digits()
    last_digit = next(digits)
    for next_digit in digits:
        if next_digit > last_digit:
            return False
        last_digit = next_digit

    # number must contain at least two consecutive digits (exactly two if strict)
    digits = generate_digits()
    last_digit = next(digits)
    consec_count = 1
    for next_digit in digits:
        if next_digit == last_digit:
            consec_count += 1
            if not strict and consec_count > 1:
                break
        else:
            if consec_count == 2:
                break
            consec_count = 1
        last_digit = next_digit
    else:
        if consec_count != 2:
            return False

    return True


@pytest.mark.parametrize(
    "n,strict,is_valid",
    (
        (111111, False, True),
        (111111, True, False),
        (223450, True, False),
        (123789, True, False),
        (112233, True, True),
        (123444, True, False),
        (111122, True, True),
    ),
)
def test_code_is_valid(n: int, strict: bool, is_valid: bool):
    assert code_is_valid(n, strict) == is_valid


if __name__ == "__main__":
    with Timer() as timer:
        main(timer)
