"""AoC 2021, Day 02: https://adventofcode.com/2021/day/2"""
from enum import Enum
from functools import reduce
from io import StringIO
from pathlib import Path
from typing import NewType, Iterable, Callable, TextIO

import numpy as np
import pytest

from util import get_input_path, Timer


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        movements = list(translate_instruction(line) for line in fp)
    timer.check("Read input")

    final_state = run_simulation(movements, ProblemPart.Part1)
    x, depth, _ = final_state[0]
    print(x * depth)
    timer.check("Part 1")

    final_state = run_simulation(movements, ProblemPart.Part2)
    x, depth, _ = final_state[0]
    print(x * depth)
    timer.check("Part 2")


# For part 1, movement vector is interpreted as ∆x, ∆y. For part 2 it is interpreted
# as displacement, ∆ aim.
Movement = NewType("Movement", np.ndarray)
# State vector consists of x, y, aim for both parts
State = NewType("State", np.ndarray)


def create_movement(displacement, delta_aim) -> Movement:
    return Movement(np.array([[displacement, delta_aim]]))


def create_state(x=0, y=0, aim=0) -> State:
    return State(np.array([[x, y, aim]]))


UNIT_MOVEMENTS = {
    "forward": create_movement(1, 0),
    "up": create_movement(0, -1),
    "down": create_movement(0, 1),
}


def move_1(initial: State, movement: Movement) -> State:
    transform = np.array(((1, 0, 0), (0, 1, 0)))
    return initial + movement.dot(transform)


def move_2(initial: State, movement: Movement) -> State:
    """
    initial = [x0, y0, a0]
    movement = [∆x, ∆a]
    final = [x0 + ∆x, y0 + ∆x * a0, a0 + ∆a]
    final = initial + [∆x, ∆x * a0, ∆a]
    [∆x, ∆x * a0, ∆a] = movement . [[1, a0, 0],
                                    [0, 0, 1]]
    """
    _, _, aim = initial[0]
    transform = np.array([[1, aim, 0], [0, 0, 1]])
    return initial + movement.dot(transform)


class ProblemPart(Enum):
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, move_fn: Callable[[State, Movement], State]):
        self.move_fn = move_fn

    Part1 = 1, move_1
    Part2 = 2, move_2


def run_simulation(movements: Iterable[Movement], part: ProblemPart) -> State:
    initial_state = create_state()
    final_state = reduce(part.move_fn, movements, initial_state)
    return final_state


def translate_instruction(instruction: str) -> Movement:
    # https://media.giphy.com/media/tyttpGTMMCADZR2YPZe/giphy.gif
    direction, magnitude = instruction.split()
    return int(magnitude) * UNIT_MOVEMENTS[direction]


TEST_INPUT = """forward 5
down 5
forward 8
up 3
down 8
forward 2
"""


@pytest.fixture
def sample_input() -> TextIO:
    with StringIO(TEST_INPUT) as fp:
        yield fp


def test_main(sample_input):
    from numpy.testing import assert_array_equal

    movements = list(translate_instruction(line) for line in sample_input)
    final_state = run_simulation(movements, ProblemPart.Part2)

    assert_array_equal(create_state(15, 60, 10), final_state)


if __name__ == "__main__":
    input_path = get_input_path(2, 2021)
    with Timer() as timer:
        main(input_path, timer)
