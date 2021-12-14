""" Advent of Code 2021, Day 13: https://adventofcode.com/2021/day/13 """
from __future__ import annotations

import sys
from dataclasses import astuple, dataclass
from enum import Enum
from io import StringIO
from itertools import takewhile
from pathlib import Path
from typing import Iterator, TextIO, Callable, Optional

import pytest

from aoc2020.aoc20 import Mask
from util import get_input_path, timer, partition

Point = tuple[int, int]


def main(input_path: Path):
    with open(input_path) as fp:
        points = read_mask(fp)
        for instruction in read_instructions(fp):
            points = points.fold(instruction)

    points.print()


class Axis(Enum):
    def __new__(cls, value: str, index: int) -> Axis:
        obj = super(Enum, cls).__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, _, index: int):
        self.axis_index = index

    X = "x", 0
    Y = "y", 1


@dataclass
class Instruction:
    axis: Axis
    value: int


class FoldableMask(Mask):
    def partition(
        self, predicate: Callable[[tuple[int, int]], bool]
    ) -> tuple[FoldableMask, FoldableMask]:
        left, right = partition(predicate, self.points)
        return FoldableMask(left), FoldableMask(right)

    def fold(self, instruction: Instruction) -> FoldableMask:
        axis, value = astuple(instruction)
        below, above = self.partition(lambda p: p[axis.axis_index] > value)
        if axis is Axis.X:
            above = above.flip_horizontal().translate(2 * value, 0)
        elif axis is Axis.Y:
            above = above.flip_vertical().translate(0, 2 * value)

        return FoldableMask(above.points | below.points)

    def print(self, out: TextIO = sys.stdout):
        printable = self.normalize()
        width, height = printable.dimensions
        for y in range(height):
            s = "".join(
                ("#" if (x, y) in printable.points else " ") for x in range(width)
            )
            out.write(s + "\n")


def read_mask(fp: TextIO) -> FoldableMask:
    point_lines = takewhile(lambda line: line != "", (line.strip() for line in fp))
    points: Iterator[Point] = (
        (int(x), int(y)) for line in point_lines for x, y in (line.split(","),)
    )
    return FoldableMask(points)


def read_next_instruction(fp: TextIO) -> Optional[Instruction]:
    return next(read_instructions(fp), None)


def read_instructions(fp: TextIO) -> Iterator[Instruction]:
    for line in fp:
        fold = line.strip().split()[-1]
        axis, value = fold.split("=")
        yield Instruction(Axis(axis), int(value))


SAMPLE_INPUT = """6,10
0,14
9,10
0,3
10,4
4,11
6,0
6,12
4,1
0,13
10,12
3,4
3,0
8,4
1,10
2,14
8,10
9,0

fold along y=7
fold along x=5
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_single_fold(sample_input):
    points = read_mask(sample_input)
    instruction = read_next_instruction(sample_input)

    points = points.fold(instruction)
    assert len(points.points) == 17


def test_folds(sample_input):
    points = read_mask(sample_input)

    for instruction in read_instructions(sample_input):
        points = points.fold(instruction)

    output = StringIO()
    points.print(output)

    expected = """#####
#   #
#   #
#   #
#####
"""
    output.seek(0)
    assert output.read() == expected


if __name__ == "__main__":
    input_path = get_input_path(13, 2021)
    with timer():
        main(input_path)
