""" Advent of Code 2021, Day 24: https://adventofcode.com/2021/day/24 """
from enum import Enum
from io import StringIO
from itertools import islice, product
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        monad = list(fp)
    timer.check("Read input")

    result = find_largest_valid_number(monad)
    print(result)
    timer.check("Part 1")


Register = tuple[int, ...]


def find_largest_valid_number(monad: Sequence[str]) -> int:
    digit_range = range(9, 0, -1)
    for model_number in product(*((digit_range,) * 14)):
        if not run_program(monad, model_number)[-1]:
            break
    result = 0
    for d in model_number:
        result *= 10
        result += d

    return result


def run_program(
        program: Iterable[str],
        stdin: Iterable[int],
        start_line: int = 0,
        initial_register: Register = (0, 0, 0, 0)
) -> Register:
    program = islice(program, start_line)
    stdin = iter(stdin)
    register = list(initial_register)

    def extract_args(args: list[str]) -> tuple[int, int]:
        """parse input as a number or get value from register"""
        reg_index = name_to_index(args[0])
        try:
            value = int(args[1])
        except ValueError:
            value = register[name_to_index(args[1])]
        return reg_index, value

    for command in program:
        instruction, *args = command.split()

        match instruction:
            case "inp":
                try:
                    register[name_to_index(args[0])] = next(stdin)
                except StopIteration:
                    print("Input exhausted.")
                    break
            case "add":
                reg_index, value = extract_args(args)
                register[reg_index] += value
            case "mod":
                reg_index, value = extract_args(args)
                register[reg_index] %= value
            case "div":
                reg_index, value = extract_args(args)
                register[reg_index] = int(register[reg_index] / value)
            case "mul":
                reg_index, value = extract_args(args)
                register[reg_index] *= value
            case "eql":
                reg_index, value = extract_args(args)
                register[reg_index] = int(register[reg_index] == value)
            case _:
                raise ValueError(f"Unrecognized instruction: '{instruction}'")

    return tuple(register)


def name_to_index(name: str) -> int:
    return ord(name) - ord("w")


SAMPLE_INPUT = """\
inp w
add z w
mod z 2
div w 2
add y w
mod y 2
div w 2
add x w
mod x 2
div w 2
mod w 2
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(
    ("value", "expected"),
    ((15, (1, 1, 1, 1)), (1, (0, 0, 0, 1)), (10, (1, 0, 1, 0)))
)
def test_sample_program(sample_input, value, expected):
    program = list(sample_input)
    result = run_program(program, [value])
    assert result == expected


if __name__ == "__main__":
    input_path = get_input_path(24, year=2021)
    with Timer() as timer:
        main(input_path, timer)