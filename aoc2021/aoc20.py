""" Advent of Code 2021, Day 20: https://adventofcode.com/2021/day/20 """

from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import pytest

from util import Timer, get_input_path

Algorithm = tuple[int, ...]

ACCUMULATION_MASK = np.array((
    (1, 2, 4),
    (8, 16, 32),
    (64, 128, 256),
))


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        algo, image = parse_input(fp)
    timer.check("Parse input")

    processed_image = apply_algorithm(image, algo, 2)
    print(processed_image.sum())
    timer.check("Part 1")

    print_image(processed_image)


def parse_input(fp: TextIO) -> tuple[Algorithm, np.ndarray]:
    algo = tuple(1 if c == "#" else 0 for c in fp.readline().strip())
    _ = fp.readline()  # throw away blank line
    img = np.array(
        tuple(tuple(1 if c == "#" else 0 for c in line.strip()) for line in fp)
    )
    return algo, img


def apply_algorithm(image: np.ndarray, algo: Algorithm, steps: int = 1, field: int = 0) -> np.ndarray:
    if steps == 0:
        return image

    padded_image = pad_image(image, 2, field)
    algo_indexes = calc_algo_indexes(padded_image)
    image = np.array(tuple(
        tuple(algo[idx] for idx in row)
        for row in algo_indexes
    ))

    match algo[0], algo[-1]:
        case 1, 1:
            next_field = 1
        case 1, 0:
            next_field = int(not field)
        case _:
            next_field = 0

    return apply_algorithm(image, algo, steps - 1, next_field)


def calc_algo_indexes(padded_image: np.ndarray) -> np.ndarray:
    x_dim, y_dim = padded_image.shape
    algo_indexes = np.zeros((x_dim + 2, y_dim + 2), dtype=int)
    for x in range(padded_image.shape[0]):
        for y in range(padded_image.shape[1]):
            algo_indexes[x:x + 3, y:y + 3] += ACCUMULATION_MASK * padded_image[x, y]
    return pad_image(algo_indexes, -2)


def pad_image(image: np.ndarray, n: int, value: int = 0) -> np.ndarray:
    if n > 0:
        return np.pad(image, ((n, n), (n, n)), constant_values=value)
    if n < 0:
        return image[-n:n, -n:n]
    return image


def print_image(image: np.ndarray):
    chars = (" ", "█")
    for row in image:
        print("".join(chars[v] for v in row))


SAMPLE_INPUT = """\
..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..##\
#..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###\
.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#.\
.#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#.....\
.#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#..\
...####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.....\
..##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#

#..#.
#....
##..#
..#..
..###
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_parse_input(sample_input):
    algo, img = parse_input(sample_input)
    expected_algo = (0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1)
    expected_img = np.array((
        (1, 0, 0, 1, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 0, 0, 1),
        (0, 0, 1, 0, 0),
        (0, 0, 1, 1, 1),
    ))

    assert len(algo) == 512
    assert algo[:len(expected_algo)] == expected_algo
    assert (img == expected_img).all()


def test_pad_image(sample_input):
    _, img = parse_input(sample_input)
    expected = np.array((
        (0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 1, 0, 0),
        (0, 1, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 1, 0),
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0),
        (0, 0, 0, 0, 0, 0, 0),
    ))
    assert (pad_image(img, 1, 0) == expected).all()
    assert (pad_image(expected, -1) == img).all()


def test_calc_algo_indexes(sample_input):
    _, img = parse_input(sample_input)
    indexes = calc_algo_indexes(pad_image(img, 2))
    assert indexes[3, 3] == 34


def test_calc_algo_indexes_2():
    image = np.array((
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ))
    expected = np.array((
        (1, 3, 7, 6, 4),
        (9, 27, 63, 54, 36),
        (73, 219, 511, 438, 292),
        (72, 216, 504, 432, 288),
        (64, 192, 448, 384, 256),
    ))
    assert (calc_algo_indexes(pad_image(image, 2)) == expected).all()


def test_apply_algo(sample_input):
    algo, img = parse_input(sample_input)
    result = apply_algorithm(img, algo)

    expected = np.array((
        (0, 1, 1, 0, 1, 1, 0),
        (1, 0, 0, 1, 0, 1, 0),
        (1, 1, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 0, 0, 1),
        (0, 1, 0, 0, 1, 1, 0),
        (0, 0, 1, 1, 0, 0, 1),
        (0, 0, 0, 1, 0, 1, 0),
    ))

    assert (result == expected).all()


def test_apply_algo_2(sample_input):
    algo, img = parse_input(sample_input)
    result = apply_algorithm(img, algo, 2)

    expected = np.array((
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (0, 1, 0, 0, 1, 0, 1, 0, 0),
        (1, 0, 1, 0, 0, 0, 1, 1, 1),
        (1, 0, 0, 0, 1, 1, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 1, 0, 1),
        (0, 1, 0, 1, 1, 1, 1, 1, 0),
        (0, 0, 1, 0, 1, 1, 1, 1, 1),
        (0, 0, 0, 1, 1, 0, 1, 1, 0),
        (0, 0, 0, 0, 1, 1, 1, 0, 0),
    ))

    assert result.sum() == 35
    assert (result == expected).all()


if __name__ == "__main__":
    input_path = get_input_path(20, year=2021)
    with Timer() as timer:
        main(input_path, timer)
