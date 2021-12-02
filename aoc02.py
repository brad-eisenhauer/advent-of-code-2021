"""Calculate the final position"""

from functools import reduce
from pathlib import Path

import numpy as np

INPUT_FILE = Path(__file__).parent / "resources" / "input02.txt"


Movement = np.ndarray  # Movement vector consists of displacement, ∆ aim
State = np.ndarray  # State vector consists of x, y, aim


def create_movement(displacement, delta_aim) -> Movement:
    return np.array([[displacement, delta_aim]])


def create_state(x=0, y=0, aim=0) -> State:
    return np.array([[x, y, aim]])


UNIT_MOVEMENTS = {
    "forward": create_movement(1, 0),
    "up": create_movement(0, -1),
    "down": create_movement(0, 1),
}


def main():
    with open(INPUT_FILE) as fp:
        movements = (translate_instruction(instruction) for instruction in fp)
        initial_state = create_state()
        final_state = reduce(move, movements, initial_state)

    print(final_state)
    x, depth, _ = final_state[0]
    print(x * depth)


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


if __name__ == "__main__":
    main()
