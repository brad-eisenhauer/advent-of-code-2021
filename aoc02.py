"""Calculate the final position"""

from functools import reduce
from pathlib import Path

import numpy as np

INPUT_FILE = Path(__file__).parent / "resources" / "input02.txt"

UNIT_MOVEMENTS = {
    "forward": np.array([[1, 0]]),
    "up": np.array([[0, -1]]),
    "down": np.array([[0, 1]]),
}


def main():
    with open(INPUT_FILE) as fp:
        movements = (create_movement(instruction) for instruction in fp)
        final_state = reduce(move, movements, np.array([[0, 0, 0]]))

    print(final_state)
    x, depth, _ = final_state[0]
    print(x * depth)


def create_movement(instruction: str) -> np.array:
    # https://media.giphy.com/media/tyttpGTMMCADZR2YPZe/giphy.gif
    direction, magnitude = instruction.split()
    return int(magnitude) * UNIT_MOVEMENTS[direction]


def move(initial: np.array, movement: np.array) -> np.array:
    """
    initial = [x0, y0, a0]
    movement = [dx, da]
    final = [x0 + dx, y0 + dx * a0, a0 + da]
    final = initial + [dx, dx * a0, da]
    [dx, dx * a0, da] = movement . [[1, a0, 0],
                                    [0, 0, 1]]
    """
    _, _, aim = initial[0]
    transform = np.array([[1, aim, 0], [0, 0, 1]])
    return initial + movement.dot(transform)


if __name__ == "__main__":
    main()
