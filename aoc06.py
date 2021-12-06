"""Advent of Code 2021, Day 06: https://adventofcode.com/2021/day/6"""
from functools import lru_cache
from io import StringIO
from typing import Iterable, Iterator, List, TextIO

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


"""This represents the daily population from a single fish, spawned ex nihilo"""
POPULATION_BY_DAY = (
    (1,) * MATURITY_INTERVAL
    + (2,) * SPAWN_INTERVAL
    + (3,) * (MATURITY_INTERVAL - SPAWN_INTERVAL)
)


def calc_population_by_day(
    days_until_spawn: Iterable[int], simulation_length: int
) -> List[int]:
    """
    The population on any given day is equal to the sum of the populations 7 and 9
    days prior.  Inspired by https://wild.maths.org/fibonacci-and-bees
    Unfortunately, it's still slower that the recursive solution above.
    """
    population_by_day = [
        sum(populations)
        for populations in zip(
            *(
                POPULATION_BY_DAY[MATURITY_INTERVAL - d : 2 * MATURITY_INTERVAL - d]
                for d in days_until_spawn
            )
        )
    ]
    for _ in range(simulation_length - MATURITY_INTERVAL):
        population_by_day.append(
            population_by_day[-MATURITY_INTERVAL] + population_by_day[-SPAWN_INTERVAL]
        )
    return population_by_day


TEST_INPUT = "3,4,3,1,2"


@pytest.mark.parametrize(("sim_days", "expected"), ((80, 5934), (256, 26984457539)))
def test_count_descendants(sim_days, expected):
    with StringIO(TEST_INPUT) as fp:
        initial_states = read_states(fp)

    result = count_all_descendants(initial_states, simulation_length=sim_days)
    assert result == expected


@pytest.mark.parametrize(("sim_days", "expected"), ((80, 5934), (256, 26984457539)))
def test_calc_population_by_day(sim_days, expected):
    with StringIO(TEST_INPUT) as fp:
        initial_states = read_states(fp)

    result = calc_population_by_day(initial_states, sim_days)
    assert result[-1] == expected


if __name__ == "__main__":
    with timer():
        main(get_input_path(6))
