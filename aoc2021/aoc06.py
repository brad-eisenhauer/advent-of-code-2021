"""Advent of Code 2021, Day 06: https://adventofcode.com/2021/day/6"""
from collections import Counter
from enum import Enum, auto
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, TextIO

import pytest

from util import get_input_path, timer

SPAWN_INTERVAL = 7
MATURITY_INTERVAL = 9


class ComputationMethod(Enum):
    def __new__(cls, value, *args, **kwargs):
        obj = super(Enum, cls).__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, _, compute: Callable[[Iterable[int], int], int]):
        self.compute = compute

    DYNAMIC = (
        auto(),
        lambda initial_states, simulation_length: calc_population_by_day(
            initial_states, simulation_length
        )[-1],
    )
    RECURSIVE = (
        auto(),
        lambda initial_states, simulation_length: count_all_descendants(
            initial_states, simulation_length
        ),
    )


def main(input_path: Path, method: ComputationMethod):
    with open(input_path) as fp:
        initial_states = read_states(fp)

    result = method.compute(initial_states, simulation_length=256)
    print(result)


def read_states(fp: TextIO) -> Iterator[int]:
    return (int(n) for n in fp.read().split(","))


def count_all_descendants(states: Iterable[int], simulation_length: int) -> int:
    return sum(count_descendants(n, simulation_length) for n in states)


@lru_cache(maxsize=256 + 2 * MATURITY_INTERVAL)
def count_descendants(days_until_spawn: int, simulation_length: int) -> int:
    """
    Count the descendants of a single fish, including the original fish

    Max cache size would be MATURITY_INTERVAL * (256 + MATURITY_INTERVAL) to account for
    every possible set in parameters, however given the current implementation, non-zero
    days_until_spawn will only ever be used with the maximum simulation_length, so the
    cache size requirement is much lower.
    """
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
    """
    days_until_spawn_counts = Counter(days_until_spawn)
    population_by_day = [
        sum(populations)
        for populations in zip(
            *(
                initial_population(d, count)
                for d, count in days_until_spawn_counts.items()
            )
        )
    ]
    for _ in range(simulation_length - MATURITY_INTERVAL):
        population_by_day.append(
            population_by_day[-MATURITY_INTERVAL] + population_by_day[-SPAWN_INTERVAL]
        )
    return population_by_day


def initial_population(days_until_spawn: int, count: int) -> Iterator[int]:
    ps = POPULATION_BY_DAY[
        MATURITY_INTERVAL - days_until_spawn : 2 * MATURITY_INTERVAL - days_until_spawn
    ]
    return (count * p for p in ps)


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
    input_path = get_input_path(6)
    with timer():
        main(input_path, ComputationMethod.DYNAMIC)
