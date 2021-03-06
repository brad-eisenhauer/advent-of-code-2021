""" Advent of Code 2021, Day 14: https://adventofcode.com/2021/day/14 """
from __future__ import annotations

import unittest
from collections import Counter
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Iterable, TextIO

import pytest

from util import get_input_path, timer, Timer


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        template, formula = read_input(fp)
    timer.check("Read input")

    result = formula.calc_element_count_range(template, 10)
    print(result)
    timer.check("Part 1")

    result = formula.calc_element_count_range(template, 40)
    print(result)
    timer.check("Part 2")


def read_input(fp: TextIO) -> tuple[str, Formula]:
    template = next(fp).strip()
    _ = next(fp)  # throw away empty line
    formula = Formula.read(fp)
    return template, formula


class Formula:
    def __init__(self, rules: Iterable[str]):
        self.rules = {k: v for k, _, v in (line.strip().split() for line in rules)}

    @classmethod
    def read(cls, fp: TextIO) -> Formula:
        return cls(fp)

    @cache
    def count_elements(self, template: str, steps: int) -> Counter:
        if steps == 0:
            return Counter(template)

        if len(template) > 2:
            # split string and count each part separately
            mid_idx = len(template) // 2
            left = template[: mid_idx + 1]
            right = template[mid_idx:]
            result = self.count_elements(left, steps) + self.count_elements(
                right, steps
            )
            # middle character is included in both counts
            result[template[mid_idx]] -= 1
            return result

        new_char = self.rules[template]
        result = self.count_elements(
            template[0] + new_char, steps - 1
        ) + self.count_elements(new_char + template[1], steps - 1)
        result[new_char] -= 1
        return result

    def calc_element_count_range(self, template: str, steps: int) -> int:
        counts = self.count_elements(template, steps)
        min_count = min(counts.values())
        max_count = max(counts.values())
        return max_count - min_count


SAMPLE_INPUT = """\
NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C
"""


@pytest.mark.parametrize(("steps", "expected"), ((10, 1588), (40, 2188189693529)))
def test_calc_element_count_range(steps, expected):
    with StringIO(SAMPLE_INPUT) as fp:
        template, formula = read_input(fp)
        result = formula.calc_element_count_range(template, steps)
        assert result == expected


if __name__ == "__main__":
    input_path = get_input_path(14, 2021)
    with Timer() as timer:
        main(input_path, timer)
