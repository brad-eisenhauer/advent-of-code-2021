"""AoC 2021, Day 02: https://adventofcode.com/2021/day/2"""
from functools import reduce
from io import StringIO
from pathlib import Path
from typing import NewType

import numpy as np

from util import get_input_path

Movement = NewType(
    "Movement", np.ndarray  # Movement vector consists of displacement, ∆ aim
)
State = NewType("State", np.ndarray)  # State vector consists of x, y, aim


def create_movement(displacement, delta_aim) -> Movement:
    return Movement(np.array([[displacement, delta_aim]]))


def create_state(x=0, y=0, aim=0) -> State:
    return State(np.array([[x, y, aim]]))


UNIT_MOVEMENTS = {
    "forward": create_movement(1, 0),
    "up": create_movement(0, -1),
    "down": create_movement(0, 1),
}


def main(input_path: Path):
    with open(input_path) as fp:
        final_state = run_simulation(fp)

    print(final_state)
    x, depth, _ = final_state[0]
    print(x * depth)


def run_simulation(input_stream):
    movements = (translate_instruction(instruction) for instruction in input_stream)
    initial_state = create_state()
    final_state = reduce(move, movements, initial_state)
    return final_state


def translate_instruction(instruction: str) -> Movement:
    # https://media.giphy.com/media/tyttpGTMMCADZR2YPZe/giphy.gif
    direction, magnitude = instruction.split()
    return int(magnitude) * UNIT_MOVEMENTS[direction]


def move(initial: State, movement: Movement) -> State:
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


TEST_INPUT = """forward 5
down 5
forward 8
up 3
down 8
forward 2
"""


def test_main():
    from numpy.testing import assert_array_equal

    with StringIO(TEST_INPUT) as fp:
        final_state = run_simulation(fp)

    assert_array_equal(create_state(15, 60, 10), final_state)


if __name__ == "__main__":
    main(get_input_path(2))
