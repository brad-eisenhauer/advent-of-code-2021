""" Advent of Code 2021, Day 11: https://adventofcode.com/2021/day/11 """

from __future__ import annotations

from io import StringIO
from itertools import product
from pathlib import Path
from typing import Iterator, TextIO, Iterable, Optional

import numpy as np
import pytest

from util import get_input_path, timer


def main(input_path: Path):
    with open(input_path) as fp:
        octopi1 = OctopusGrid.read(fp)

    octopi2 = octopi1.copy()

    # Part 1
    result = octopi1.simulate(100)
    print(result)

    # Part 2
    print(octopi2.run_until_synchronized())


class OctopusGrid:
    def __init__(self, energy: Iterable[Iterable[int]]):
        energy = tuple((0, *row, 0) for row in energy)
        grid_size = len(energy)
        null_row = (0,) * (grid_size + 2)
        self.energy = np.array((null_row, *energy, null_row))
        self.count = grid_size ** 2

        # Create matrix of ones with zero borders
        self.increment_matrix = np.ones_like(self.energy, dtype=int)
        self.increment_matrix[0, :] = 0
        self.increment_matrix[:, 0] = 0
        self.increment_matrix[-1, :] = 0
        self.increment_matrix[:, -1] = 0

    @classmethod
    def read(cls, fp: TextIO) -> OctopusGrid:
        return cls((int(n) for n in line.strip()) for line in fp)

    def copy(self) -> OctopusGrid:
        return OctopusGrid(self.energy[1:-1, 1:-1])

    def tick(self) -> int:
        # Increment all energies by 1
        self.energy += self.increment_matrix

        # All octopi whose energy is greater than nine flash increasing the energy of
        # adjacent octopi
        has_flashed = self.calc_flashes()
        # has_flashed, energy_increase = self.calc_flash_energy()
        # self.energy += energy_increase

        # Reset the energy of all flashed octopi to 0
        self.energy *= ~has_flashed
        self.energy *= self.increment_matrix  # reset borders to zero

        return has_flashed.sum()

    def calc_flash_energy(
        self, flashes: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the energy increase matrix

        If the energy increase from the current set of flashes causes no additional
        octopi to flash, then the flashes are stable. Otherwise try again with the
        additional flashes.

        Parameters
        ----------
        flashes
            Boolean matrix indicating the locations of octopi which will flash. If
            omitted, will use the current energy state.

        Returns
        -------
        Flashes and energy released
        """
        if flashes is None:
            flashes = self.energy > 9

        flash_energy = flashes.astype(int)
        flash_energy += np.roll(flash_energy, shift=1, axis=0) + np.roll(
            flash_energy, shift=-1, axis=0
        )
        flash_energy += np.roll(flash_energy, shift=1, axis=1) + np.roll(
            flash_energy, shift=-1, axis=1
        )
        flash_energy *= self.increment_matrix

        new_flashes = (self.energy + flash_energy) > 9
        if (flashes == new_flashes).all():
            return flashes, flash_energy
        return self.calc_flash_energy(new_flashes)

    def calc_flashes(self) -> np.ndarray:

        def generate_neighbors(x, y) -> Iterator[tuple[int, int]]:
            yield x - 1, y - 1
            yield x - 1, y
            yield x - 1, y + 1
            yield x, y - 1
            yield x, y + 1
            yield x + 1, y - 1
            yield x + 1, y
            yield x + 1, y + 1

        x_range = range(1, self.energy.shape[0] - 1)
        y_range = range(1, self.energy.shape[1] - 1)

        queue = []
        visited = set()

        for x, y in product(x_range, y_range):
            if self.energy[x, y] > 9:
                queue.append((x, y))

        flashed = np.zeros_like(self.energy, dtype=bool)
        while len(queue) > 0:
            point = queue.pop(0)
            if point in visited:
                continue
            visited.add(point)

            x, y = point
            flashed[x, y] = True
            for adj_x, adj_y in generate_neighbors(x, y):
                self.energy[adj_x, adj_y] += 1
                # Border octopi cannot reach energy > 3 in a single tick, so we don't
                # need to check for those.
                if self.energy[adj_x, adj_y] > 9:
                    queue.append((adj_x, adj_y))

        return flashed

    def simulate(self, ticks: int) -> int:
        return sum(self.tick() for _ in range(ticks))

    def run_until_synchronized(self) -> int:
        step_count = 1
        while True:
            if self.tick() == self.count:
                break
            step_count += 1
        return step_count


SAMPLE_INPUT = """5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_simulate(sample_input):
    octopi = OctopusGrid.read(sample_input)
    assert octopi.simulate(100) == 1656


def test_synchronize(sample_input):
    octopi = OctopusGrid.read(sample_input)
    assert octopi.run_until_synchronized() == 195


if __name__ == "__main__":
    input_path = get_input_path(11)
    with timer():
        main(input_path)
