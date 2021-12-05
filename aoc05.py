"""Advent of Code 2021, Day 05: https://adventofcode.com/2021/day/5"""

from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from util import get_input_path, greatest_common_divisor

Point = Tuple[int, ...]


def main(input_path: Path):
    with open(input_path) as fp:
        counter = count_overlaps(translate_to_segments(fp), orthogonal_only=False)
        result = sum(1 for p in counter if counter[p] > 1)
        print(result)


def translate_to_segments(lines: Iterable[str]) -> Iterator[Tuple[Point, Point]]:
    for line in lines:
        yield make_start_end(line)


def count_overlaps(
    segments: Iterable[Tuple[Point, Point]], orthogonal_only: bool = False
) -> Counter:
    return Counter(
        p
        for segment in segments
        for p in generate_overlapping_points(*segment, orthogonal_only)
    )


def make_start_end(line_segment: str) -> Tuple[Point, Point]:
    start, _, end = line_segment.split()
    return tuple(map(int, start.split(","))), tuple(map(int, end.split(",")))


def generate_overlapping_points(
    start: Point, end: Point, orthogonal_only: bool
) -> Iterator[Point]:
    delta = subtract_points(end, start)
    if orthogonal_only and all(p != 0 for p in delta):
        return
    unit_delta = int_divide_point(delta, greatest_common_divisor(*delta))

    point = start
    yield point
    while point != end:
        point = add_points(point, unit_delta)
        yield point


# region Point arithmetic


def add_points(a: Point, b: Point) -> Point:
    return tuple(n + m for n, m in zip(a, b))


def subtract_points(a: Point, b: Point) -> Point:
    return tuple(n - m for n, m in zip(a, b))


def multiply_point(p: Point, m: int) -> Point:
    return tuple(n * m for n in p)


def int_divide_point(p: Point, d: int) -> Point:
    return tuple(n // d for n in p)


# endregion


TEST_INPUT = """0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2
"""


def test_count_orthogonal_overlaps():
    with StringIO(TEST_INPUT) as fp:
        counter = count_overlaps(translate_to_segments(fp), orthogonal_only=True)

    result = sum(1 for p in counter if counter[p] > 1)

    assert 5 == result


def test_count_all_overlaps():
    with StringIO(TEST_INPUT) as fp:
        counter = count_overlaps(translate_to_segments(fp), orthogonal_only=False)

    result = sum(1 for p in counter if counter[p] > 1)

    assert 12 == result


if __name__ == "__main__":
    main(get_input_path(5))
