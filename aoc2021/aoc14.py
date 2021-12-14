""" Advent of Code 2021, Day 14: https://adventofcode.com/2021/day/14 """

from collections import Counter
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO

import pytest

from util import get_input_path, timer

Formula = dict[str, str]


def main(input_path: Path):
    with open(input_path) as fp:
        template, formula = read_input(fp)

    counts = count_elements(template, formula, 40)
    diff = max(counts.values()) - min(counts.values())
    print(diff)


def read_input(fp: TextIO) -> tuple[str, Formula]:
    template = next(fp).strip()
    _ = next(fp)  # throw away empty line
    formula = {k: v for k, _, v in (line.strip().split() for line in fp)}
    return template, formula


def count_elements(template: str, formula: Formula, steps: int) -> Counter:
    @cache
    def count_elements_rec(partial_template: str, steps: int) -> Counter:
        if steps == 0:
            return Counter(partial_template)

        if len(partial_template) > 2:
            # split string and count each half separately
            mid_idx = len(partial_template) // 2
            left = partial_template[:mid_idx + 1]
            right = partial_template[mid_idx:]
            result = count_elements_rec(left, steps) + count_elements_rec(right, steps)
            # middle character is included in both counts
            result[partial_template[mid_idx]] -= 1
            return result

        new_char = formula[partial_template]
        result = (
            count_elements_rec(partial_template[0] + new_char, steps - 1)
            + count_elements_rec(new_char + partial_template[1], steps - 1)
        )
        result[new_char] -= 1
        return result

    return count_elements_rec(template, steps)


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


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(("step_count", "expected"), ((10, 1588), (40, 2188189693529)))
def test_most_least_common_elements(sample_input, step_count, expected):
    template, formula = read_input(sample_input)
    counts = count_elements(template, formula, step_count)
    min_count = min(counts.values())
    max_count = max(counts.values())
    assert max_count - min_count == expected


if __name__ == "__main__":
    input_path = get_input_path(14, 2021)
    with timer():
        main(input_path)
