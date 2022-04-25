""" Advent of Code 2019, Day 03: https://adventofcode.com/2019/day/3 """
import operator
from functools import reduce
from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO, Iterable

import pytest

from util import Timer, get_input_path

Vector = tuple[int, ...]
Route = Iterable[str]

UNIT_VECTORS = {"L": (-1, 0), "R": (1, 0), "U": (0, 1), "D": (0, -1)}


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        routes = fp.readlines()
    timer.check("Read input")

    nearest_intersection_distance = calc_nearest_intersection_distance(routes)
    print(nearest_intersection_distance)
    timer.check("Part 1")

    min_wire_distance = calc_min_total_wire_distance_to_intersection(routes)
    print(min_wire_distance)
    timer.check("Part 2")


def get_points_covered_by_route(route: str) -> dict[Vector, int]:
    pos = (0, 0)
    result = {}
    step_count = 0
    for instruction in route.strip().split(","):
        direction = instruction[0]
        for _ in range(int(instruction[1:])):
            pos = tuple(a + b for a, b in zip(pos, UNIT_VECTORS[direction]))
            step_count += 1
            if pos not in result:
                result[pos] = step_count

    return result


def calc_nearest_intersection_distance(routes: Iterable[Route]) -> int:
    point_sets = (get_points_covered_by_route(r) for r in routes)
    intersections = reduce(
        operator.and_, (point_set.keys() for point_set in point_sets)
    )
    result = min(sum(abs(v) for v in intersection) for intersection in intersections)
    return result


def calc_min_total_wire_distance_to_intersection(routes: Iterable[Route]) -> int:
    left, right = (get_points_covered_by_route(r) for r in routes)
    intersections = left.keys() & right.keys()
    result = min(left[i] + right[i] for i in intersections)
    return result


SAMPLE_INPUT = [
    """\
R75,D30,R83,U83,L12,D49,R71,U7,L72
U62,R66,U55,R34,D71,R55,D58,R83
""",
    """\
R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51
U98,R91,D20,R16,D67,R40,U7,R15,U6,R7
""",
]


@pytest.mark.parametrize(("sample_index", "expected"), ((0, 159), (1, 135)))
def test_calc_nearest_intersection_distance(sample_index, expected):
    routes = (line for line in StringIO(SAMPLE_INPUT[sample_index]))
    result = calc_nearest_intersection_distance(routes)
    assert result == expected


@pytest.mark.parametrize(("sample_index", "expected"), ((0, 610), (1, 410)))
def test_cal_min_total_wire_distance_to_intersection(sample_index, expected):
    routes = (line for line in StringIO(SAMPLE_INPUT[sample_index]))
    result = calc_min_total_wire_distance_to_intersection(routes)
    assert result == expected


if __name__ == "__main__":
    input_path = get_input_path(3, year=2019)
    with Timer() as timer:
        main(input_path, timer)
