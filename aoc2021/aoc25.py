""" Advent of Code 2021, Day 25: https://adventofcode.com/2021/day/25 """
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO, Optional, Iterable

import pytest

from util import Timer, get_input_path

Vector = tuple[int, ...]


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        grid = Grid.read(fp)
    timer.check("Read input")

    result = grid.step_until_frozen()
    print(result)
    timer.check("Part 1")


class Orientation(Enum):
    def __new__(cls, *args, **kwargs) -> Orientation:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, target: Vector):
        self.target = target

    EAST = ">", (0, 1)
    SOUTH = "v", (1, 0)


@dataclass
class SeaCucumber:
    position: Vector
    orientation: Orientation
    grid: Grid

    def next_position(self) -> Vector:
        return tuple(
            (p + t) % d
            for p, t, d in zip(
                self.position, self.orientation.target, self.grid.dimensions
            )
        )

    def can_move(self) -> bool:
        return not self.grid.is_occupied(self.next_position())


class Grid:
    def __init__(self, dimensions: Vector):
        self.dimensions = dimensions
        self.grid: list[list[Optional[SeaCucumber]]] = [
            [None for _ in range(dimensions[1])] for _ in range(dimensions[0])
        ]
        self.sea_cucumbers_by_orientation: dict[
            Orientation, list[SeaCucumber]
        ] = defaultdict(list)

    @classmethod
    def read(cls, fp: TextIO) -> Grid:
        lines = tuple(line.strip() for line in fp)
        result = cls(dimensions=(len(lines), len(lines[0])))
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if char in ("v", ">"):
                    result.add_sea_cucumber((row, col), Orientation(char))
        return result

    def is_occupied(self, position: Vector) -> bool:
        return self.get_position(position) is not None

    def add_sea_cucumber(self, position: Vector, orientation: Orientation):
        if self.is_occupied(position):
            raise ValueError(f"Position {position} is already occupied.")
        sea_cucumber = SeaCucumber(position, orientation, self)
        r, c = position
        self.grid[r][c] = sea_cucumber
        self.sea_cucumbers_by_orientation[sea_cucumber.orientation].append(sea_cucumber)

    def get_position(self, position: Vector) -> Optional[SeaCucumber]:
        r, c = position
        return self.grid[r][c]

    def set_position(self, position: Vector, sea_cucumber: Optional[SeaCucumber]):
        r, c = position
        self.grid[r][c] = sea_cucumber

    def move_sea_cucumber(self, sea_cucumber: SeaCucumber):
        target = sea_cucumber.next_position()
        if self.is_occupied(target):
            raise ValueError(f"Target position is already occupied.")
        if sea_cucumber is not self.get_position(sea_cucumber.position):
            raise ValueError(f"Different sea cucumber at the origin position.")
        self.set_position(sea_cucumber.position, None)
        self.set_position(target, sea_cucumber)
        sea_cucumber.position = target

    def move_all(self, sea_cucumbers: Iterable[SeaCucumber]) -> int:
        result = 0
        for sc in sea_cucumbers:
            self.move_sea_cucumber(sc)
            result += 1
        return result

    def step(self) -> int:
        result = 0  # total number of sea cucumbers that moved
        for orientation in Orientation:
            movers = tuple(
                c
                for c in self.sea_cucumbers_by_orientation[orientation]
                if c.can_move()
            )
            result += len(movers)
            self.move_all(movers)

        return result

    def step_until_frozen(self) -> int:
        result = 1
        while self.step() > 0:
            result += 1
        return result

    def print(self, output: TextIO = sys.stdout):
        output.write(self.build_print_string())

    def build_print_string(self) -> str:
        lines = []
        for row in self.grid:
            line = "".join("." if sc is None else sc.orientation.value for sc in row)
            lines.append(line + "\n")
        return "".join(lines)


SAMPLE_INPUT = """\
v...>>.vv>
.vv>>.vv..
>>.>v>...v
>>v>>.>.v.
v>v.vv.v..
>.>>..v...
.vv..>.>v.
v.v..>>v.v
....v..v.>
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


INITIAL_STEPS = [
    """\
....>.>v.>
v.v>.>v.v.
>v>>..>v..
>>v>v>.>.v
.>v.v...v.
v>>.>vvv..
..v...>>..
vv...>>vv.
>.v.v..v.v
""",
    """\
>.v.v>>..v
v.v.>>vv..
>v>.>.>.v.
>>v>v.>v>.
.>..v....v
.>v>>.v.v.
v....v>v>.
.vv..>>v..
v>.....vv.
""",
    """\
v>v.v>.>v.
v...>>.v.v
>vv>.>v>..
>>v>v.>.v>
..>....v..
.>.>v>v..v
..v..v>vv>
v.v..>>v..
.v>....v..
"""
]


def test_read_grid(sample_input):
    grid = Grid.read(sample_input)
    assert grid.build_print_string() == SAMPLE_INPUT


def test_step(sample_input):
    grid = Grid.read(sample_input)
    for i in range(len(INITIAL_STEPS)):
        grid.step()
        assert grid.build_print_string() == INITIAL_STEPS[i]


def test_run_until_frozen(sample_input):
    grid = Grid.read(sample_input)
    result = grid.step_until_frozen()
    assert result == 58


if __name__ == "__main__":
    input_path = get_input_path(25, year=2021)
    with Timer() as timer:
        main(input_path, timer)
