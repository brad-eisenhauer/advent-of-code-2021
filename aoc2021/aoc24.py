""" Advent of Code 2021, Day 24: https://adventofcode.com/2021/day/24 """
import heapq
import operator
from enum import Enum
from io import StringIO
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO, Optional, Callable

import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        monad = list(fp)
    timer.check("Read input")

    result = find_largest_valid_number(monad)
    print(result)
    timer.check("Part 1")

    result = find_smallest_valid_number(monad)
    print(result)
    timer.check("Part 2")


Program = Sequence[str]
ProgramIndex = int
Register = tuple[int, ...]
Result = int
State = tuple[ProgramIndex, Result]


class SearchTarget(Enum):
    Minimum = "min", range(9, 0, -1), operator.lt
    Maximum = "max", range(1, 10), operator.gt

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, digit_range: range, comparator: Callable[[int, int], bool]):
        self.digit_range = digit_range
        self.comparator = comparator


def find_largest_valid_number(validator: Program) -> Optional[int]:
    return search_implementation(validator, SearchTarget.Maximum)


def find_smallest_valid_number(validator: Program) -> Optional[int]:
    return search_implementation(validator, SearchTarget.Minimum)


def search_implementation(validator: Program, target: SearchTarget) -> Optional[int]:
    """
    Search for valid values

    This implementation is a depth-first search. It runs the validator program
    incrementally, recording the state after each input digit. "State" in this case
    encompasses the index of the next program segment to run and the output value of
    the 'z' register. Examination of the MONAD program demonstrates that the other
    registers are never relevant to the output of any program segment.

    When searching for the maximum valid value, the output of each program segment is
    calculated for each input in ascending order. This leaves the output from the
    largest digit on top of the stack, which means the largest total input values will
    be the next processed. Effectively, we're counting down from all '9's. When
    searching for the minimum value, the processing order is reversed.
    """
    # Divide the program so that we can process each digit individually and retain each
    # state after processing.
    sub_progs = divide_program(validator)
    initial_state = (0, 0)
    values: dict[State, int] = {initial_state: 0}  # max/min value leading to each state
    states: list[State] = [initial_state]  # stack of states to continue processing
    goal_state = (len(sub_progs), 0)  # We're looking for 0 output after the final step

    while len(states) > 0:
        state = states.pop()

        if state == goal_state:
            return values[state]

        prog_index, z_value = state
        if prog_index >= len(sub_progs):
            continue

        for n in target.digit_range:
            prog_result = run_program(
                sub_progs[prog_index], [n], initial_register=(0, 0, 0, z_value)
            )
            new_state = (prog_index + 1, prog_result[-1])
            new_value = values[state] * 10 + n

            if (
                new_state not in values
                or target.comparator(new_value, values[new_state])
            ):
                values[new_state] = new_value
                states.append(new_state)

    return None


def divide_program(program: Program) -> Sequence[Program]:
    """
    Split a program into sub-programs, such that each starts with an 'inp' instruction
    """
    result = []
    next_sub_prog = []
    for line in program:
        if line.startswith("inp"):
            result.append(next_sub_prog)
            next_sub_prog = [line]
        else:
            next_sub_prog.append(line)
    result.append(next_sub_prog)
    return result[1:]


def run_program(
        program: Iterable[str],
        stdin: Iterable[int],
        start_line: int = 0,
        stop_line: Optional[int] = None,
        initial_register: Register = (0, 0, 0, 0)
) -> Register:
    """Run the specified program, or portion of a program"""
    program = islice(program, start_line, stop_line)
    stdin = iter(stdin)
    register = list(initial_register)

    def extract_args(args: list[str]) -> tuple[int, int]:
        # parse input as a number or get value from register
        reg_index = name_to_index(args[0])
        try:
            value = int(args[1])
        except ValueError:
            value = register[name_to_index(args[1])]
        return reg_index, value

    def name_to_index(name: str) -> int:
        return ord(name) - ord("w")

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


SAMPLE_INPUT = [
    """\
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
""",
    # last 3 "steps" of real program
    """\
inp w
mul x 0
add x z
mod x 26
div z 1
add x 10
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 1
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -6
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 10
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -8
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 3
mul y x
add z y
"""
]


@pytest.fixture
def sample_program() -> Program:
    with StringIO(SAMPLE_INPUT[0]) as fp:
        program = list(fp)
    return program


@pytest.fixture
def short_validator() -> Program:
    with StringIO(SAMPLE_INPUT[1]) as fp:
        program = list(fp)
    return program


@pytest.mark.parametrize(
    ("value", "expected"),
    ((15, (1, 1, 1, 1)), (1, (0, 0, 0, 1)), (10, (1, 0, 1, 0)))
)
def test_sample_program(sample_program, value, expected):
    result = run_program(sample_program, [value])
    assert result == expected


@pytest.mark.parametrize(
    ("stdin", "expected"),
    (
        ((1, 2, 3), (3, 1, 6, 6)),
        ((1, 1, 3), (3, 0, 0, 0)),
        ((9, 7, 9), (9, 0, 0, 0)),
    ))
def test_longer_sample_program(short_validator, stdin, expected):
    result = run_program(short_validator, stdin)
    assert result == expected


def test_find_largest_valid_number(short_validator):
    result = find_largest_valid_number(short_validator)
    assert result == 979


def test_find_smallest_valid_number(short_validator):
    result = find_smallest_valid_number(short_validator)
    assert result == 113


if __name__ == "__main__":
    input_path = get_input_path(24, year=2021)
    with Timer() as timer:
        main(input_path, timer)