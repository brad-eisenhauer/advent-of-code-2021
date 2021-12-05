"""Advent of Code 2021, Day 05: https://adventofcode.com/2021/day/5"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pytest

from util import get_input_path, greatest_common_divisor


def main(input_path: Path):
    with open(input_path) as fp:
        intersection_counter = count_intersections(
            translate_to_segments(fp), orthogonal_only=False
        )

    result = sum(1 for p in intersection_counter if intersection_counter[p] > 1)
    print(result)


@dataclass(frozen=True)
class Vector:
    coords: Tuple[int, ...]

    def __add__(self, other: Vector) -> Vector:
        return Vector.from_iterable(a + b for a, b in zip(self.coords, other.coords))

    def __mul__(self, other: int) -> Vector:
        return Vector.from_iterable(x * other for x in self.coords)

    def __floordiv__(self, other: int) -> Vector:
        return Vector.from_iterable(x // other for x in self.coords)

    def __sub__(self, other: Vector) -> Vector:
        return Vector.from_iterable(a - b for a, b in zip(self.coords, other.coords))

    @property
    def is_orthogonal(self) -> bool:
        return sum(1 for c in self.coords if c != 0) < 2

    @classmethod
    def from_string(cls, coords: str) -> Vector:
        return cls.from_iterable(int(x) for x in coords.split(","))

    @classmethod
    def from_iterable(cls, coords: Iterable[int]) -> Vector:
        return cls(tuple(coords))


def translate_to_segments(lines: Iterable[str]) -> Iterator[Tuple[Vector, Vector]]:
    """Map input lines to line segments"""
    for line in lines:
        start, _, end = line.split()
        yield Vector.from_string(start), Vector.from_string(end)


def count_intersections(
    segments: Iterable[Tuple[Vector, Vector]], orthogonal_only: bool = False
) -> Counter[Vector, int]:
    """Count the number of line segments intersecting each point"""
    return Counter(
        p
        for start, end in segments
        for p in generate_points_on_line_segment(start, end, orthogonal_only)
    )


def generate_points_on_line_segment(
    start: Vector, end: Vector, orthogonal_only: bool
) -> Iterator[Vector]:
    """Generate all the discreet points through which the given segment passes"""
    delta = end - start
    if orthogonal_only and not delta.is_orthogonal:
        return
    increment = delta // greatest_common_divisor(*delta.coords)

    point = start
    yield point
    while point != end:
        point += increment
        yield point


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


@pytest.mark.parametrize(("orthogonal_only", "expected"), ((True, 5), (False, 12)))
def test_count_overlaps(orthogonal_only, expected):
    with StringIO(TEST_INPUT) as fp:
        counter = count_intersections(translate_to_segments(fp), orthogonal_only)

    result = sum(1 for p in counter if counter[p] > 1)

    assert expected == result


if __name__ == "__main__":
    main(get_input_path(5))
