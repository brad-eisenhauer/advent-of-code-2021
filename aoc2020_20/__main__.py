"""Advent of Code 2020, Day 20: https://adventofcode.com/2020/day/20"""
from __future__ import annotations

import logging
import operator
import time
from functools import reduce
from io import StringIO
from itertools import chain, product, takewhile
from math import sqrt
from typing import Iterable, Iterator, Mapping, Optional, Sequence, TextIO

from aoc2020_20 import Edge, Image, Mask, Tile
from util import get_input_path, make_sequence, timer

log = logging.getLogger("2020_20")

SEA_MONSTER = """
                  # 
#    ##    ##    ###
 #  #  #  #  #  #   
"""


def main(input_data: TextIO):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    log.info("Reading input data...")
    tiles = read_tiles(input_data)
    log.info("Assembling image...")
    image = assemble_image(tiles)
    corner_product = reduce(
        operator.mul, (image[row][col].id for row, col in product((0, -1), (0, -1)))
    )
    log.info("Image assembled. Product of corner tile IDs == %d", corner_product)

    log.info("Searching for sea monsters...")
    monster_mask = get_monster_mask()
    monsters = make_sequence(find_sea_monsters(image, monster_mask))
    log.info("Found %d sea monsters.", len(monsters))

    log.info("Calculating water roughness...")
    roughness = calc_water_roughness(image, monsters)
    log.info("Water roughness == %d", roughness)

    return roughness


def print_time():
    print(f"{time.time()}: ", end="")


def read_tiles(fp: TextIO) -> Iterator[Tile]:
    try:
        while True:
            id_line = fp.readline()
            tile_id = int(id_line.split()[1][:-1])
            tile_data = takewhile(lambda s: s.strip() != "", fp)
            yield Tile(tile_id, tile_data)
    except (IOError, ValueError, IndexError) as e:
        ...


def assemble_image(tiles: Iterable[Tile], image: Image = None) -> Optional[Image]:
    tiles = make_sequence(tiles)

    if not tiles:
        if image is not None:
            return image if image.is_complete else None
        return None

    if image is None:
        size = int(sqrt(len(tiles)))
        image = Image.new(size)

    # find first unfilled location
    image_height, image_width = image.dimensions
    for row_idx, col_idx in product(range(image_height), range(image_width)):
        if image[row_idx][col_idx] is None:
            break

    edge_requirements = get_edge_requirements(image, row_idx, col_idx)
    for tile in find_corner_tiles(tiles) if (row_idx, col_idx) == (0, 0) else tiles:
        for candidate_match in tile.find_matches(edge_requirements):
            result = assemble_image(
                (t for t in tiles if t is not tile),
                image.insert(candidate_match, row_idx, col_idx),
            )
            if result is not None:
                return result

    return None


def get_edge_requirements(
    image: Image, row_idx: int, col_idx: int
) -> Mapping[Edge, int]:
    adjacent_offsets = {
        Edge.TOP: (-1, 0),
        Edge.BOTTOM: (1, 0),
        Edge.LEFT: (0, -1),
        Edge.RIGHT: (0, 1),
    }
    edge_requirements = {}
    idxs = (row_idx, col_idx)
    for edge in Edge:
        if (
            adj_tile := image.get(
                *(idx + offset for idx, offset in zip(idxs, adjacent_offsets[edge]))
            )
        ) is not None:
            edge_requirements[edge] = adj_tile.edge_hashes[edge.opposite]
    return edge_requirements


def find_corner_tiles(tiles: Sequence[Tile]) -> Iterator[Tile]:
    """
    Corner tile is any tile with exactly two adjacent edges whose hashes cannot be
    matched by any other tile.  Tile will be rotated or flipped so unmatched edges
    are top and left.
    """
    for tile in tiles:
        unmatched_edges = []
        for edge in Edge:
            edge_requirements = {edge.opposite: tile.edge_hashes[edge]}
            for adj_tile in tiles:
                if (
                    adj_tile is not tile
                    and next(adj_tile.find_matches(edge_requirements), None) is not None
                ):
                    break
            else:
                unmatched_edges.append(edge)
        if len(unmatched_edges) == 2 and Edge.are_adjacent(*unmatched_edges):
            if Edge.TOP not in unmatched_edges:
                tile = tile.flip_vertical()
            if Edge.LEFT not in unmatched_edges:
                tile = tile.flip_horizontal()
            yield tile


def get_monster_mask():
    """It was a sea-floor smash."""
    with StringIO(SEA_MONSTER) as sm:
        return Mask(
            (x, y) for x, line in enumerate(sm)
            for y, char in enumerate(line)
            if char == "#"
        )


def find_sea_monsters(image: Image, monster_mask: Mask) -> Iterator[Mask]:
    image_data = make_sequence(image.merge())
    image_dims = len(image_data), len(image_data[0])
    for mask_variation in monster_mask.generate_variations():
        mask_dims = mask_variation.dimensions
        x_range = range(image_dims[0] - mask_dims[0] + 1)
        y_range = range(image_dims[1] - mask_dims[1] + 1)
        for x_offset, y_offset in product(x_range, y_range):
            candidate_mask = mask_variation.translate(x_offset, y_offset)
            if does_mask_match(image_data, candidate_mask):
                yield candidate_mask


def does_mask_match(image_data: Sequence[str], mask: Mask) -> bool:
    return all(image_data[x][y] == "#" for x, y in mask.points)


def calc_water_roughness(image: Image, monsters: Iterable[Mask]) -> int:
    monster_points = set(chain.from_iterable(m.points for m in monsters))
    result = sum(
        1
        for x, row in enumerate(image.merge())
        for y, char in enumerate(row)
        if char == "#" and (x, y) not in monster_points
    )
    return result


TEST_INPUT = """Tile 2311:
..##.#..#.
##..#.....
#...##..#.
####.#...#
##.##.###.
##...#.###
.#.#.#..##
..#....#..
###...#.#.
..###..###

Tile 1951:
#.##...##.
#.####...#
.....#..##
#...######
.##.#....#
.###.#####
###.##.##.
.###....#.
..#.#..#.#
#...##.#..

Tile 1171:
####...##.
#..##.#..#
##.#..#.#.
.###.####.
..###.####
.##....##.
.#...####.
#.##.####.
####..#...
.....##...

Tile 1427:
###.##.#..
.#..#.##..
.#.##.#..#
#.#.#.##.#
....#...##
...##..##.
...#.#####
.#.####.#.
..#..###.#
..##.#..#.

Tile 1489:
##.#.#....
..##...#..
.##..##...
..#...#...
#####...#.
#..#.#.#.#
...#.#.#..
##.#...##.
..##.##.##
###.##.#..

Tile 2473:
#....####.
#..#.##...
#.##..#...
######.#.#
.#...#.#.#
.#########
.###.#..#.
########.#
##...##.#.
..###.#.#.

Tile 2971:
..#.#....#
#...###...
#.#.###...
##.##..#..
.#####..##
.#..####.#
#..#.#..#.
..####.###
..#.#.###.
...#.#.#.#

Tile 2729:
...#.#.#.#
####.#....
..#.#.....
....#..#.#
.##..##.#.
.#.####...
####.#.#..
##.####...
##..#.##..
#.##...##.

Tile 3079:
#.#.#####.
.#..######
..#.......
######....
####.#..#.
.#...#.##.
#.#####.##
..#.###...
..#.......
..#.###...
"""


def test_main():
    with StringIO(TEST_INPUT) as fp:
        result = main(fp)
    assert 273 == result


if __name__ == "__main__":
    with timer():
        with open(get_input_path(20, year=2020)) as fp:
            result = main(fp)

    print(result)
