from __future__ import annotations

from enum import Enum, auto
from functools import cached_property


class Edge(Enum):
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

    @staticmethod
    def are_adjacent(a: Edge, b: Edge) -> bool:
        return not Edge.are_opposite(a, b)

    @staticmethod
    def are_opposite(a: Edge, b: Edge) -> bool:
        return a.opposite is b

    @cached_property
    def opposite(self):
        return {
            Edge.TOP: Edge.BOTTOM,
            Edge.BOTTOM: Edge.TOP,
            Edge.LEFT: Edge.RIGHT,
            Edge.RIGHT: Edge.LEFT,
        }[self]
