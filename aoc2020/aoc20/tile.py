from __future__ import annotations

from functools import cached_property
from typing import Iterable, Iterator, Mapping

from .edge import Edge


class Tile:
    def __init__(self, id: int, img_data: Iterable[str]):
        self.id = id
        self.img_data = tuple(s.strip() for s in img_data)
        self.size = len(self.img_data)
        if self.size == 0:
            raise ValueError("Empty tile data.")
        if any(len(s) != self.size for s in self.img_data):
            raise ValueError("Image data too short.")

    @cached_property
    def edges(self) -> Mapping[Edge, str]:
        return {
            Edge.TOP: self.img_data[0],
            Edge.BOTTOM: self.img_data[-1],
            Edge.LEFT: "".join(s[0] for s in self.img_data),
            Edge.RIGHT: "".join(s[-1] for s in self.img_data),
        }

    def rotate_left(self) -> Tile:
        new_data = (
            "".join(s[i - 1] for s in self.img_data) for i in range(self.size, 0, -1)
        )
        return Tile(self.id, new_data)

    def rotate_right(self) -> Tile:
        new_data = (
            "".join(s[i] for s in reversed(self.img_data)) for i in range(self.size)
        )
        return Tile(self.id, new_data)

    def flip_horizontal(self) -> Tile:
        new_data = (s[::-1] for s in self.img_data)
        return Tile(self.id, new_data)

    def flip_vertical(self) -> Tile:
        new_data = self.img_data[::-1]
        return Tile(self.id, new_data)

    def find_matches(self, edge_requirements: Mapping[Edge, str]) -> Iterator[Tile]:
        """Find transformations of the current Tile that satisfy the given edges"""
        for tile in self.generate_transformations():
            tile_edges = tile.edges
            if all(tile_edges[edge] == e for edge, e in edge_requirements.items()):
                yield tile

    def generate_transformations(self) -> Iterator[Tile]:
        """Generate reflections and rotations of the Tile"""
        next_variation = self
        for _ in range(4):
            yield next_variation
            yield next_variation.flip_vertical()
            next_variation = next_variation.rotate_right()

    def strip_edges(self) -> Iterator[str]:
        return (s[1:-1] for s in self.img_data[1:-1])
