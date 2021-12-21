""" Advent of Code 2019, Day 01: https://adventofcode.com/2019/day/1 """

from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO, Iterable

import pytest

from util import get_input_path, Timer


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        masses = list(parse_input(fp))
    timer.check("Parse input")

    result_1 = sum(calc_fuel_requirement(m) for m in masses)
    print(result_1)
    timer.check("Part 1")

    result_2 = sum(calc_total_fuel(m) for m in masses)
    print(result_2)
    timer.check("Part 2")


def parse_input(fp: TextIO) -> Iterator[int]:
    for line in fp:
        yield int(line.strip())


def calc_fuel_requirement(mass: int) -> int:
    return max(0, mass // 3 - 2)


def calc_total_fuel(mass: int) -> int:
    added_fuel = calc_fuel_requirement(mass)
    result = added_fuel
    while (added_fuel := calc_fuel_requirement(added_fuel)) > 0:
        result += added_fuel
    return result


SAMPLE_INPUT = """\
12
14
1969
100756
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(
    ("mass", "expected"), ((12, 2), (14, 2), (1969, 654), (100756, 33583))
)
def test_calc_fuel_requirement(mass, expected):
    assert calc_fuel_requirement(mass) == expected


@pytest.mark.parametrize(
    ("mass", "expected"), ((12, 2), (14, 2), (1969, 966), (100756, 50346))
)
def test_calc_total_fuel(mass, expected):
    assert calc_total_fuel(mass) == expected


if __name__ == "__main__":
    input_path = get_input_path(1, year=2019)
    with Timer() as timer:
        main(input_path, timer)
