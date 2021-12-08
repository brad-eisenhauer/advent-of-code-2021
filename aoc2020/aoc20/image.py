from __future__ import annotations

from itertools import product
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from .edge import Edge
from .tile import Tile


class Image:
    """Two-dimensional array of tiles"""

    def __init__(self, tiles: Iterable[Iterable[Optional[Tile]]]):
        self.tiles = tuple(tuple(row) for row in tiles)

    def __getitem__(self, index):
        return self.tiles[index]

    def __len__(self):
        return len(self.tiles)

    def merge(self) -> Iterator[str]:
        """Strip tile edges and merge into a single tile"""
        if not self.is_complete:
            raise ValueError("Cannot merge an incomplete image.")
        for tile_row in self.tiles:
            for tile_lines in zip(*(tile.strip_edges() for tile in tile_row)):
                yield "".join(tile_lines)

    def get(self, row: int, col: int) -> Optional[Tile]:
        try:
            return self[row][col]
        except IndexError:
            return None

    def insert(self, tile: Tile, row: int, col: int) -> Image:
        row_range = range(min(0, row), len(self.tiles))
        col_range = range(min(0, col), len(self.tiles[0]))
        new_data = (
            (
                tile
                if row_idx == row and col_idx == col
                else self.get(row_idx, col_idx)
                for col_idx in col_range
            )
            for row_idx in row_range
        )
        return Image(new_data)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return len(self.tiles), len(self.tiles[0])

    @property
    def is_complete(self) -> bool:
        return all(all(tile is not None for tile in row) for row in self)

    @property
    def is_empty(self) -> bool:
        return all(all(tile is None for tile in row) for row in self)

    @classmethod
    def new(cls, size: int) -> Image:
        return cls(((None,) * size,) * size)

    def locations(self) -> Iterator[Tuple[int, int]]:
        return product(range(len(self.tiles)), range(len(self.tiles[0])))

    def calc_edge_requirements(self, row: int, col: int) -> Mapping[Edge, str]:
        """Calculate the required hashes for each edge bordering an existing Tile"""
        adjacent_offsets = {
            Edge.TOP: (-1, 0),
            Edge.BOTTOM: (1, 0),
            Edge.LEFT: (0, -1),
            Edge.RIGHT: (0, 1),
        }
        edge_requirements = {}
        for edge in Edge:
            if (
                adj_tile := self.get(
                    *(
                        idx + offset
                        for idx, offset in zip((row, col), adjacent_offsets[edge])
                    )
                )
            ) is not None:
                edge_requirements[edge] = adj_tile.edges[edge.opposite]
        return edge_requirements
