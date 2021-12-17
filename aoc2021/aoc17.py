""" Advent of Code 2021, Day 17: https://adventofcode.com/2021/day/17 """

import re
from io import StringIO
from math import ceil, sqrt
from pathlib import Path
from typing import Iterator, Optional, TextIO

import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        target = fp.readline()
    timer.check("Read input")

    x_range, y_range = parse_target_area(target)
    print(calc_max_height(y_range))
    timer.check("Part 1")

    launch_vectors = calc_launch_vectors(x_range, y_range)
    print(len(launch_vectors))
    timer.check("Part 2")


def parse_target_area(target: str) -> tuple[range, range]:
    number_pattern = re.compile(r"[xy]=(-?\d+)\.\.(-?\d+)")
    x_match = number_pattern.search(target)
    y_match = number_pattern.search(target, pos=x_match.span()[-1] + 1)
    x_min, x_max = x_match.groups()
    y_min, y_max = y_match.groups()
    return range(int(x_min), int(x_max) + 1), range(int(y_min), int(y_max) + 1)


def calc_max_height(y_range: range) -> int:
    downward_v = min(y_range)
    initial_v = abs(downward_v) - 1
    height = initial_v * (initial_v + 1) // 2
    return height


def calc_launch_vectors(x_range: range, y_range: range) -> set[tuple[int, int]]:
    """
    Calculate unique launch vectors for which the trajectory passes through the target
    area.
    """
    results = set()
    for viy, t in generate_initial_y_velocities(y_range):
        for vix in generate_initial_x_velocities(x_range, t):
            results.add((vix, viy))
    return results


def generate_initial_y_velocities(y_range: range) -> Iterator[tuple[int, int]]:
    """
    Find all initial y-velocities that will put the probe in the target range at some
    point.

    Output is initial y-velocity and number of steps to reach target.
    """
    hi_y = max(y_range)
    minimum_viy = min(y_range)
    for viy in range(minimum_viy, -minimum_viy):
        t = calc_t(viy, hi_y)
        while calc_d(viy, t) in y_range:
            yield viy, t
            t += 1


def generate_initial_x_velocities(x_range: range, t: int) -> Iterator[int]:
    """
    Find all x-velocities that can hit the target range in t steps.

    Distance given initial velocity will be:

    """
    vi = calc_vi(min(x_range), t, True)
    while calc_d(vi, t, True) in x_range:
        yield vi
        vi += 1


def calc_d(vi: int, t: int, stop_at_max: bool = False) -> int:
    """
    Implements: d = vi * t + a * t * (t - 1) / 2

    For all cases in the current problem, a == -1, so the above simplifies to:
        d = vi*t - t * (t - 1) // 2

    Additionally, maximum distance occurs when t == vi (when v reaches 0). For this
    case the above simplifies to:
        d = vi * (vi + 1) //2
    """
    if stop_at_max and t >= vi:
        return vi * (vi + 1) // 2
    d = vi * t
    d -= t * (t - 1) // 2
    return d


def calc_vi(d: int, t: int, stop_at_max: bool = False) -> int:
    """
    Calc minimum initial velocity (vi) required to reach at least d displacement in
    time t. Again, this presumes a = -1.
    """
    vi = ceil(d / t + (t - 1) / 2)
    if stop_at_max and t > vi:
        # t is past peak distance; find minimum vi with peak displacement at least d
        return ceil(-0.5 + sqrt(0.25 + 2 * d))
    return vi


def calc_t(vi: int, d: int) -> Optional[int]:
    """
    Calculate the minimum t at which displacement is at least d. If displacement cannot
    reach d, return None.

    d = vi*t - t * (t - 1) // 2
    0 = -1/2 * t**2 + (vi + 1/2) * t - d

    t = quadratic formula, where:
        a == -1/2
        b == vi + 1/2
        c == -disp
        4ac == 2 * d
        2a == -1
    """
    disciminant = (vi + 1 / 2) ** 2 - 2 * d
    if disciminant < 0:
        return None

    t1 = (vi + 0.5) - sqrt(disciminant)
    t2 = (vi + 0.5) + sqrt(disciminant)

    for t in (t1, t2):
        if t >= 0:
            return ceil(t)
    return None


SAMPLE_INPUT = """\
target area: x=20..30, y=-10..-5
"""

SAMPLE_VECTORS = """\
23,-10  25,-9   27,-5   29,-6   22,-6   21,-7   9,0     27,-7   24,-5
25,-7   26,-6   25,-5   6,8     11,-2   20,-5   29,-10  6,3     28,-7
8,0     30,-6   29,-8   20,-10  6,7     6,4     6,1     14,-4   21,-6
26,-10  7,-1    7,7     8,-1    21,-9   6,2     20,-7   30,-10  14,-3
20,-8   13,-2   7,3     28,-8   29,-9   15,-3   22,-5   26,-8   25,-8
25,-6   15,-4   9,-2    15,-2   12,-2   28,-9   12,-3   24,-6   23,-7
25,-10  7,8     11,-3   26,-7   7,1     23,-9   6,0     22,-10  27,-6
8,1     22,-8   13,-4   7,6     28,-6   11,-4   12,-4   26,-9   7,4
24,-10  23,-8   30,-8   7,0     9,-1    10,-1   26,-5   22,-9   6,5
7,5     23,-6   28,-10  10,-2   11,-1   20,-9   14,-2   29,-7   13,-3
23,-5   24,-8   27,-9   30,-7   28,-5   21,-10  7,9     6,6     21,-5
27,-10  7,2     30,-9   21,-8   22,-7   24,-9   20,-6   6,9     29,-5
8,-2    27,-8   30,-5   24,-7
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def sample_vectors() -> set[tuple[int, int]]:
    return {
        (int(x), int(y))
        for line in StringIO(SAMPLE_VECTORS)
        for vector_str in line.split()
        for x, y in (vector_str.split(","),)
    }


def test_parse_target_area(sample_input: TextIO):
    x_range, y_range = parse_target_area(sample_input.readline())
    assert x_range == range(20, 31)
    assert y_range == range(-10, -4)


def test_calc_max_height(sample_input: TextIO):
    _, y_range = parse_target_area(sample_input.readline())
    assert calc_max_height(y_range) == 45


def test_calc_launch_vectors(sample_input: TextIO):
    x_range, y_range = parse_target_area(sample_input.readline())
    vectors = calc_launch_vectors(x_range, y_range)
    assert vectors == sample_vectors()


def test_generate_initial_y_velocities():
    y_range = range(-7, -4)
    expected = {
        (-7, 1),
        (-6, 1),
        (-5, 1),
        (-3, 2),
        (-2, 2),
        (-1, 3),
        (0, 4),
        (1, 5),
        (2, 7),
        (4, 10),
        (5, 12),
        (6, 14),
    }
    assert set(generate_initial_y_velocities(y_range)) == expected


@pytest.mark.parametrize(
    ("vi", "t", "limit", "expected"),
    ((4, 3, False, 9), (4, 6, False, 9), (4, 6, True, 10), (-1, 3, False, -6)),
)
def test_calc_d(vi, t, limit, expected):
    assert calc_d(vi, t, limit) == expected


@pytest.mark.parametrize(
    ("vi", "d", "expected"), ((3, 3, 1), (3, 4, 2), (3, 5, 2), (3, 6, 3), (3, 7, None))
)
def test_calc_t(vi, d, expected):
    assert calc_t(vi, d) == expected


@pytest.mark.parametrize(
    ("d", "t", "limit", "expected"),
    ((6, 2, False, 4), (12, 3, False, 5), (0, 1, False, 0), (20, 9, True, 6)),
)
def test_calc_vi(d, t, limit, expected):
    assert calc_vi(d, t, limit) == expected


if __name__ == "__main__":
    input_path = get_input_path(17, year=2021)
    with Timer() as t:
        main(input_path, t)
