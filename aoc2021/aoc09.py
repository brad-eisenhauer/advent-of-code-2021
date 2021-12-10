""" Advent of Code 2021, Day 09: https://adventofcode.com/2021/day/9 """
from __future__ import annotations

import operator
from enum import Enum
from functools import cached_property, lru_cache, reduce
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO

import pytest

from util import get_input_path, timer


class Direction(tuple, Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @cached_property
    def opposite(self):
        return {
            self.UP: self.DOWN,
            self.DOWN: self.UP,
            self.LEFT: self.RIGHT,
            self.RIGHT: self.LEFT,
        }[self]

    def from_point(self, point: tuple[int, int]) -> tuple[int, int]:
        row, col = point
        row_offset, col_offset = self.value
        return row + row_offset, col + col_offset


class HeightMap:
    def __init__(self, heights: Iterable[Iterable[int]]):
        self.heights = tuple((9, *row, 9) for row in heights)
        map_width = len(self.heights[0])
        self.heights = ((9,) * map_width, *self.heights, (9,) * map_width)

    @classmethod
    def read(cls, fp: TextIO) -> HeightMap:
        return cls((int(n) for n in line.strip()) for line in fp)

    def get_height(self, point: tuple[int, int]) -> int:
        row, col = point
        return self.heights[row][col]

    def find_low_points(self) -> Iterator[tuple[int, int]]:
        row_range = range(1, len(self.heights) - 1)
        col_range = range(1, len(self.heights[0]) - 1)

        found_low_point = [[False] * len(row) for row in self.heights]

        for point in product(row_range, col_range):
            if found_low_point[point[0]][point[1]] or self.get_height(point) == 9:
                continue
            if (low_point := self.find_low_point(point)) is not None:
                yield low_point
                for row, col in self.find_basin_points(low_point):
                    found_low_point[row][col] = True

    def find_low_point(self, start: tuple[int, int]) -> Optional[tuple[int, int]]:
        value = self.get_height(start)
        if value == 9:
            return None

        try:
            downstream_point = next(
                p
                for d in Direction
                for p in (d.from_point(start),)
                if self.get_height(p) < value
            )
            return self.find_low_point(downstream_point)
        except StopIteration:
            return start

    def find_upstream_points(
        self, start: tuple[int, int], acc: Optional[set[int]] = None
    ) -> set[tuple[int, int]]:
        """
        Start with the current point in the accumulator.
        For each surrounding point:
            - If the point is already in the accumulator, skip it.
            - If any flows from the adjacent point flow back to the current point, add
                the adjacent point to the accumulator, and call find_basin_points with
                the adjacent point as the starting point.
        """
        if acc is None:
            acc = {start}

        value = self.get_height(start)
        for direction in Direction:
            adj_point = direction.from_point(start)
            if adj_point in acc:
                continue
            adj_value = self.get_height(adj_point)
            if value < adj_value < 9:
                acc = self.find_upstream_points(adj_point, acc | {adj_point})

        return acc

    def find_basin_sizes(self) -> Iterator[int]:
        map_height = len(self.heights)
        map_width = len(self.heights[0])

        is_basin = list([False] * map_width for _ in range(map_height))

        for point in product(range(map_height), range(map_width)):
            row, col = point
            if not is_basin[row][col] and self.get_height(point) < 9:
                basin_size = 0
                for row, col in self.find_basin_points(point):
                    basin_size += 1
                    is_basin[row][col] = True
                yield basin_size

    def find_basin_points(self, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
        """Non-recursive BFS alternative to find_upstream_points"""
        queue = [start]
        visited = set()

        while len(queue) > 0:
            if (point := queue.pop(0)) in visited:
                continue
            visited.add(point)
            if self.get_height(point) < 9:
                yield point
                queue.extend(d.from_point(point) for d in Direction)


def main(input_path: Path):
    with open(input_path) as fp:
        height_map = HeightMap.read(fp)

    ranked_basin_sizes = sorted(height_map.find_basin_sizes(), reverse=True)
    result = reduce(operator.mul, ranked_basin_sizes[:3])

    print(result)


TEST_INPUT = """2199943210
3987894921
9856789892
8767896789
9899965678
"""


@pytest.fixture
def height_map():
    with StringIO(TEST_INPUT) as fp:
        height_map = HeightMap.read(fp)
    return height_map


def test_calc_sum_of_risks(height_map):
    low_points = height_map.find_low_points()
    result = sum(height_map.get_height(p) + 1 for p in low_points)
    assert result == 15


def test_sum_3_largest_basins(height_map):
    # low_points = height_map.find_low_points()
    # basin_sizes = (
    #     sum(1 for _ in height_map.find_basin_points(low_point))
    #     for low_point in low_points
    # )
    ranked_basin_sizes = sorted(height_map.find_basin_sizes(), reverse=True)
    result = reduce(operator.mul, ranked_basin_sizes[:3])

    assert result == 1134


if __name__ == "__main__":
    with timer():
        main(get_input_path(9))
