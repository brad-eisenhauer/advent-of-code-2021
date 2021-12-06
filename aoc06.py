"""Advent of Code 2021, Day 06: https://adventofcode.com/2021/day/6"""
from functools import lru_cache
from io import StringIO
from typing import TextIO

import pytest

from util import get_input_path


def main(input_path):
    with open(input_path) as fp:
        initial_state = read_state(fp)

    result = run_simulation(initial_state, days=256)

    print(result)


def read_state(fp: TextIO) -> list[int]:
    return [int(n) for n in fp.read().split(",")]


def run_simulation(state: list[int], days: int) -> int:
    return sum(simulate_single_fish(n, days) for n in state)


@lru_cache(maxsize=9 * 257)
def simulate_single_fish(days_until_spawn: int, simulation_length: int) -> int:
    if simulation_length < 1:
        return 1
    if days_until_spawn == 0:
        return (
            simulate_single_fish(6, simulation_length - 1)
            + simulate_single_fish(8, simulation_length - 1)
        )
    # Jump forward to next spawn date
    return simulate_single_fish(0, simulation_length - days_until_spawn)


TEST_INPUT = "3,4,3,1,2"


@pytest.mark.parametrize(("sim_days", "expected"), ((80, 5934), (256, 26984457539)))
def test_count_lanternfish(sim_days, expected):
    with StringIO(TEST_INPUT) as fp:
        initial_state = read_state(fp)

    result = run_simulation(initial_state, days=sim_days)

    assert expected == result


if __name__ == "__main__":
    main(get_input_path(6))
