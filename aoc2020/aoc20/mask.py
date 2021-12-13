from __future__ import annotations

from functools import cached_property
from typing import Iterable, Iterator, Tuple


class Mask:
    def __init__(self, points: Iterable[Tuple[int, int]]):
        self.points: set[Tuple[int, int]] = set(points)

    @cached_property
    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        min_x = min(x for x, _ in self.points)
        max_x = max(x for x, _ in self.points)
        min_y = min(y for _, y in self.points)
        max_y = max(y for _, y in self.points)
        return (min_x, min_y), (max_x, max_y)

    @cached_property
    def dimensions(self) -> Tuple[int, int]:
        (min_x, min_y), (max_x, max_y) = self.bounds
        return max_x - min_x + 1, max_y - min_y + 1

    def normalize(self) -> Mask:
        (x_offset, y_offset), _ = self.bounds
        return self.translate(-x_offset, -y_offset)

    def rotate_left(self) -> Mask:
        return Mask((-x, y) for x, y in self.transpose())

    def rotate_right(self) -> Mask:
        return Mask((x, -y) for x, y in self.transpose())

    def flip_horizontal(self) -> Mask:
        return Mask((-x, y) for x, y in self.points)

    def flip_vertical(self) -> Mask:
        return Mask((x, -y) for x, y in self.points)

    def translate(self, x_offset: int, y_offset: int) -> Mask:
        return Mask((x + x_offset, y + y_offset) for x, y in self.points)

    def transpose(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.points:
            yield y, x

    def generate_transformations(self) -> Iterator[Mask]:
        """Generate rotations and reflections of the Mask"""
        next_variation = self
        for _ in range(4):
            yield next_variation.normalize()
            yield next_variation.flip_vertical().normalize()
            next_variation = next_variation.rotate_right()
