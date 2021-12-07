""" Advent of Code 2021, Day 07: https://adventofcode.com/2021/day/7 """
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, TextIO, Tuple

import numpy as np
import pytest

from util import get_input_path, timer


class FuelConsumptionMode(Enum):
    def __new__(cls, value: str, *args, **kwargs):
        obj = super(Enum, cls).__new__(cls)
        obj._value_ = value
        return obj

    def __init__(
        self,
        _,
        fuel_consumption_fn: Callable[[int], int],
        optimization_fn: Callable[[Sequence[int]], int],
    ):
        self.fuel_consumption_fn = fuel_consumption_fn
        self.optimization_fn = optimization_fn

    LINEAR = auto(), lambda d: d, np.median
    QUADRATIC = (
        auto(),
        lambda d: d * (d + 1) // 2,
        lambda ps: round(np.average(ps)),
    )


def main(input_path: Path):
    with open(input_path) as fp:
        initial_positions = read_positions(fp)
        optimal_position, fuel_consumption = optimize_positions(
            initial_positions, FuelConsumptionMode.QUADRATIC
        )

    print(f"Found optimal position: {optimal_position}")
    print(f"Total fuel consumed: {fuel_consumption}")


def read_positions(fp: TextIO) -> Sequence[int]:
    return [int(n) for n in fp.read().strip().split(",")]


def optimize_positions(
    initial_positions: Sequence[int],
    fuel_mode: FuelConsumptionMode,
    initial_guess: Optional[int] = None,
) -> Tuple[int, int]:
    """Find local minimum of fuel consumption"""
    if initial_guess is None:
        initial_guess = fuel_mode.optimization_fn(initial_positions)

    left_fuel_consumption, fuel_consumption, right_fuel_consumption = (
        calc_fuel_consumption(initial_positions, initial_guess + offset, fuel_mode)
        for offset in (-1, 0, 1)
    )
    if fuel_consumption <= min(left_fuel_consumption, right_fuel_consumption):
        return initial_guess, fuel_consumption
    if left_fuel_consumption < right_fuel_consumption:
        return optimize_positions(initial_positions, fuel_mode, initial_guess - 1)
    return optimize_positions(initial_positions, fuel_mode, initial_guess + 1)


def calc_fuel_consumption(
    initial_positions: Iterable[int],
    final_position: int,
    fuel_mode: FuelConsumptionMode,
) -> int:
    return sum(
        fuel_mode.fuel_consumption_fn(abs(p - final_position))
        for p in initial_positions
    )


TEST_INPUT = "16,1,2,0,4,2,7,1,2,14"


@pytest.mark.parametrize(
    ("fuel_mode", "expected"),
    ((FuelConsumptionMode.LINEAR, 37), (FuelConsumptionMode.QUADRATIC, 168)),
)
def test_optimize(fuel_mode, expected):
    with StringIO(TEST_INPUT) as fp:
        initial_positions = read_positions(fp)
        optimal_position, fuel_consumption = optimize_positions(
            initial_positions, fuel_mode
        )

    assert fuel_consumption == expected


if __name__ == "__main__":
    with timer():
        main(get_input_path(7))
