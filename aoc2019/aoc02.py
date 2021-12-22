""" Advent of Code 2019, Day 02: https://adventofcode.com/2019/day/2 """

from io import StringIO
from itertools import product
from pathlib import Path
from typing import Iterator, TextIO, Optional

import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        orig_buffer = read_intcode_buffer(fp)
    timer.check("Read input")

    buffer = orig_buffer.copy()
    buffer[1] = 12
    buffer[2] = 2
    run_intcode(buffer)
    print(buffer[0])
    timer.check("Part 1")

    for noun, verb in product(range(100), range(100)):
        buffer = orig_buffer.copy()
        buffer[1:3] = noun, verb
        run_intcode(buffer)
        if buffer[0] == 19690720:
            break

    print(100 * noun + verb)
    timer.check("Part 2")


def read_intcode_buffer(fp: TextIO) -> list[int]:
    return list(map(int, fp.readline().split(",")))


def run_intcode(buffer: list[int]):
    pointer = 0
    while (pointer := process_opcode(buffer, pointer)) is not None:
        ...


def process_opcode(buffer: list[int], pointer: int) -> Optional[int]:
    match buffer[pointer]:
        case 1:
            left_idx, right_idx, result_idx = buffer[pointer + 1: pointer + 4]
            buffer[result_idx] = buffer[left_idx] + buffer[right_idx]
            next_pointer = pointer + 4
        case 2:
            left_idx, right_idx, result_idx = buffer[pointer + 1: pointer + 4]
            buffer[result_idx] = buffer[left_idx] * buffer[right_idx]
            next_pointer = pointer + 4
        case 99:
            next_pointer = None
        case _:
            raise ValueError(f"Unrecognized opcode ({buffer[pointer]}) at index {pointer}.")

    return next_pointer


SAMPLE_INPUT = """\
1,9,10,3,2,3,11,0,99,30,40,50
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_read_intcode_buffer(sample_input):
    assert read_intcode_buffer(sample_input) == [1, 9, 10, 3, 2, 3, 11, 0, 99, 30, 40, 50]


@pytest.mark.parametrize(
    ("pointer", "expected_result", "expected_buffer"),
    (
        (0, 4, [1, 9, 10, 70, 2, 3, 11, 0, 99, 30, 40, 50]),
        (4, 8, [150, 9, 10, 3, 2, 3, 11, 0, 99, 30, 40, 50])
    )
)
def test_process_opcode(sample_input, pointer, expected_result, expected_buffer):
    buffer = read_intcode_buffer(sample_input)
    next_pointer = process_opcode(buffer, pointer)
    assert next_pointer == expected_result
    assert buffer == expected_buffer


@pytest.mark.parametrize(("buffer", "expected_buffer"), (
    ([1,0,0,0,99], [2,0,0,0,99]),
    ([2,3,0,3,99], [2,3,0,6,99]),
    ([2,4,4,5,99,0], [2,4,4,5,99,9801]),
    ([1,1,1,4,99,5,6,0,99], [30,1,1,4,2,5,6,0,99])
))
def test_run_intcode(buffer, expected_buffer):
    run_intcode(buffer)
    assert buffer == expected_buffer


if __name__ == "__main__":
    input_path = get_input_path(2, year=2019)
    with Timer() as timer:
        main(input_path, timer)
