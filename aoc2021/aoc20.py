""" Advent of Code 2021, Day 20: https://adventofcode.com/2021/day/20 """

from io import StringIO
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import pytest

from util import Timer, get_input_path

Algorithm = tuple[int, ...]


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        algo, image = parse_input(fp)
    timer.check("Parse input")

    processed_image = apply_algorithm(image, algo, 2)
    print(processed_image.sum())
    timer.check("Part 1")

    processed_image = apply_algorithm(processed_image, algo, 48)
    print(processed_image.sum())
    timer.check("Part 2")


def parse_input(fp: TextIO) -> tuple[Algorithm, np.ndarray]:
    algo = tuple(1 if c == "#" else 0 for c in fp.readline().strip())
    _ = fp.readline()  # throw away blank line
    img = np.array(
        tuple(tuple(1 if c == "#" else 0 for c in line.strip()) for line in fp)
    )
    return algo, img


def apply_algorithm(
    image: np.ndarray, algo: Algorithm, iterations: int, field: int = 0
) -> np.ndarray:
    """
    Apply the given algorithm to the image.

    Resulting image will be two elements per iteration larger than the original image
    in each dimension.

    Parameters
    ----------
    image
        Two-dimensional binary integer array
    algo
        Algorithm to apply
    iterations
        Number of iterations of the algorithm to apply
    field
        Value of pixels in the infinite surroundings

    Returns
    -------
    The processed image as a two-dimensional ndarray
    """
    match algo[0], algo[-1]:
        case 1, 1:
            calc_new_field_value = lambda _: 1
        case 1, 0:
            calc_new_field_value = lambda fv: int(not fv)
        case _:
            calc_new_field_value = lambda _: 0

    for _ in range(iterations):
        # Pad the initial image on all sides with two layers of the field value. One
        # layer will be the additional size of the result image. The second will inform
        # the resulting values in the additional layer.
        padded_image = pad_image(image, 2, field)
        algo_indexes = calc_algo_indexes(padded_image)
        image = np.array(tuple(
            tuple(algo[idx] for idx in row)
            for row in algo_indexes
        ))

        field = calc_new_field_value(field)

    return image


# contribution of each pixel to its surrounding pixels
ACCUMULATION_MATRIX = np.array((
    (1, 2, 4),
    (8, 16, 32),
    (64, 128, 256),
))


def calc_algo_indexes(padded_image: np.ndarray) -> np.ndarray:
    """
    Calculate the algorithm indexes for each pixel of the new image.

    Parameters
    ----------
    padded_image
        Original image with field padding; final matrix will be two elements smaller in
        each dimension than the padded image.

    Returns
    -------
    Algorithm indexes as a two-dimensional ndarray
    """
    # Working matrix, algo_indexes, is two elements larger in each dimension than the
    # padded image. The final result will be the range [2:-2, 2:-2] from this matrix,
    # which is two elements smaller than the padded image. The additional size is added
    # to avoid bounds-checking below.
    x_dim, y_dim = padded_image.shape
    algo_indexes = np.zeros((x_dim + 2, y_dim + 2), dtype=int)
    for x in range(x_dim):
        for y in range(y_dim):
            # Add values contributed by padded_image[x, y] to the matrix of indexes.
            # Note position padded_image[x, y] corresponds to algo_indexes[x+1, y+1].
            algo_indexes[x : x + 3, y : y + 3] += ACCUMULATION_MATRIX * padded_image[x, y]
    # Clip working matrix to dimensions of final image.
    return algo_indexes[2:-2, 2:-2]


def pad_image(image: np.ndarray, n: int, value: int = 0) -> np.ndarray:
    """
    Pad all sides of the array with n rows of the specified value.

    Parameters
    ----------
    image
        Two-dimensional ndarray
    n
        Number of rows to add
    value
        Contents of the added rows

    Returns
    -------
    Padded image as two-dimensional ndarray
    """
    return np.pad(image, ((n, n), (n, n)), constant_values=value)


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
    result = apply_algorithm(img, algo, iterations=1)

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
    result = apply_algorithm(img, algo, iterations=2)

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


def test_apply_algo_50(sample_input):
    algo, img = parse_input(sample_input)
    result = apply_algorithm(img, algo, iterations=50)
    assert result.sum() == 3351


if __name__ == "__main__":
    input_path = get_input_path(day=20, year=2021)
    with Timer() as timer:
        main(input_path, timer)
