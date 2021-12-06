"""Advent of Code 2021, Day 06: https://adventofcode.com/2021/day/6"""
from functools import lru_cache
from io import StringIO
from typing import Iterable, Iterator, TextIO, Union

import pytest

from util import get_input_path, timer

SPAWN_INTERVAL = 7
MATURITY_INTERVAL = 9


def main(input_path):
    with open(input_path) as fp:
        initial_states = read_states(fp)

    result = count_all_descendants(initial_states, simulation_length=256)
    print(result)


def read_states(fp: TextIO) -> Iterator[int]:
    return (int(n) for n in fp.read().split(","))


def count_all_descendants(states: Iterable[int], simulation_length: int) -> int:
    return sum(count_descendants(n, simulation_length) for n in states)


@lru_cache(maxsize=MATURITY_INTERVAL * (256 + MATURITY_INTERVAL))
def count_descendants(days_until_spawn: int, simulation_length: int) -> int:
    if simulation_length < 1:
        return 1
    if days_until_spawn == 0:
        return (
            # Jump forward to next spawn dates
            count_descendants(0, simulation_length - SPAWN_INTERVAL)
            + count_descendants(0, simulation_length - MATURITY_INTERVAL)
        )
    return count_descendants(0, simulation_length - days_until_spawn)


TEST_INPUT = "3,4,3,1,2"


@pytest.mark.parametrize(("sim_days", "expected"), ((80, 5934), (256, 26984457539)))
def test_count_descendants(sim_days, expected):
    with StringIO(TEST_INPUT) as fp:
        initial_states = read_states(fp)

    result = count_all_descendants(initial_states, simulation_length=sim_days)
    assert expected == result


if __name__ == "__main__":
    with timer():
        main(get_input_path(6))
