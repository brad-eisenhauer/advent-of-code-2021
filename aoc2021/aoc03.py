"""AoC 2021, Day 03: https://adventofcode.com/2021/day/3"""
import unittest
from functools import partial
from io import StringIO
from typing import Callable, Iterable

import pandas as pd

from util import get_input_path, timer


def main(input_file):
    with open(input_file) as fp:
        diagnostic_readings = read_diagnostics(fp)

    power_consumption = calc_power_consumption(diagnostic_readings)
    life_support_rating = calc_life_support_rating(diagnostic_readings)

    print(f"Power consumption: {power_consumption}")
    print(f"Life support rating: {life_support_rating}")


def read_diagnostics(diagnostic_output: Iterable[str]) -> pd.DataFrame:
    """Transform input to data frame; one column per digit"""
    return pd.DataFrame(list(line.strip()) for line in diagnostic_output)


def calc_power_consumption(diagnostic_readings: pd.DataFrame) -> int:
    gamma_rate_str = "".join(diagnostic_readings.mode().iloc[0])
    gamma_rate = int(gamma_rate_str, 2)
    mask = int("1" * len(gamma_rate_str), 2)
    epsilon_rate = gamma_rate ^ mask

    return gamma_rate * epsilon_rate


def calc_life_support_rating(diagnostic_readings: pd.DataFrame) -> int:
    o2_rating = calc_life_support_component(diagnostic_readings, filter_most_common)
    co2_rating = calc_life_support_component(
        diagnostic_readings, partial(filter_most_common, invert=True)
    )

    return o2_rating * co2_rating


def calc_life_support_component(
    diagnostic_readings: pd.DataFrame,
    filter_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
    col: int = 0,
) -> int:
    """Calculate O2 or CO2 rating, depending on the filter_fn"""
    if len(diagnostic_readings) == 1:
        return int("".join(diagnostic_readings.iloc[0]), 2)

    filtered_readings = filter_fn(diagnostic_readings, col)
    return calc_life_support_component(filtered_readings, filter_fn, col + 1)


def filter_most_common(
    diagnostic_readings: pd.DataFrame, col: int, invert: bool = False
) -> pd.DataFrame:
    """Select rows containing the most (or least) common value in the specified column"""
    most_common = diagnostic_readings[col] == diagnostic_readings[col].mode().iloc[-1]
    return diagnostic_readings[~most_common if invert else most_common]


TEST_INPUT = """00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010
"""


class Day03Tests(unittest.TestCase):
    def setUp(self):
        self.readings = read_diagnostics(StringIO(TEST_INPUT))

    def test_power_consumption(self):
        result = calc_power_consumption(self.readings)
        assert 198 == result

    def test_life_support(self):
        result = calc_life_support_rating(self.readings)
        assert 230 == result


if __name__ == "__main__":
    input_path = get_input_path(3, 2021)
    with timer():
        main(input_path)
