""" Advent of Code 2021, Day 15: https://adventofcode.com/2021/day/15 """
from dataclasses import astuple, dataclass
from heapq import heappop, heappush
from io import StringIO
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import networkx as nx
import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        risks = read_risks(fp)
    timer.check("Read input")

    for expanded in (False, True):
        graph = build_grid(risks, expanded)
        timer.check(f"Build grid - expanded={expanded}")

        start = Point(0, 0)
        goal = Point(
            len(risks) * (EXPANSION_MULTIPLIER if expanded else 1) - 1,
            len(risks[0]) * (EXPANSION_MULTIPLIER if expanded else 1) - 1,
        )
        result = find_minimum_risk_path(graph, start, goal)
        print(result)
        timer.check(f"Calc path - expanded={expanded}")


EXPANSION_MULTIPLIER = 5


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __gt__(self, other):
        return astuple(self) > astuple(other)


def read_risks(fp: TextIO) -> Sequence[Sequence[int]]:
    return [list(map(int, line.strip())) for line in fp]


def build_grid(risks: Sequence[Sequence[int]], expand: bool = False) -> nx.DiGraph:
    def build_line(index: int) -> Sequence[int]:
        base_line = risks[index % len(risks)]
        if expand:
            result = [
                (n - 1 + repetition + index // len(risks)) % 9 + 1
                for repetition in range(EXPANSION_MULTIPLIER)
                for n in base_line
            ]
            return result
        return base_line

    def generate_all_edges() -> Iterator[tuple[Point, Point, int]]:
        prev_line = build_line(0)
        yield from generate_edges_in_line(0, prev_line)
        for x in range(
            1, len(risks) if not expand else len(risks) * EXPANSION_MULTIPLIER
        ):
            this_line = build_line(x)
            yield from generate_edges_in_line(x, this_line)
            for y, (this_cost, prev_cost) in enumerate(zip(this_line, prev_line)):
                this_pt = Point(x, y)
                prev_pt = Point(x - 1, y)
                yield this_pt, prev_pt, prev_cost
                yield prev_pt, this_pt, this_cost

    def generate_edges_in_line(
        x: int, costs: Sequence[int]
    ) -> Iterator[tuple[Point, Point, int]]:
        for y, (left_cost, right_cost) in enumerate(zip(costs, costs[1:])):
            left_pt = Point(x, y)
            right_pt = Point(x, y + 1)
            yield left_pt, right_pt, right_cost
            yield right_pt, left_pt, left_cost

    g = nx.DiGraph()
    g.add_weighted_edges_from(generate_all_edges(), "cost")

    return g


def find_minimum_risk_path(graph: nx.DiGraph, start: Point, end: Point = None) -> int:
    """
    A* algorithm based on https://www.redblobgames.com/pathfinding/a-star/introduction.html
    """
    if end is None:
        end = max(graph.nodes)

    frontier = []
    heappush(frontier, (0, start))

    def heuristic(point: Point) -> int:
        return abs(end.x - point.x) + abs(end.y - point.y)

    came_from = {start: None}
    cost_so_far = {start: 0}

    while len(frontier) > 0:
        _, current = heappop(frontier)
        if current == end:
            break

        for next in graph.successors(current):
            new_cost = cost_so_far[current] + graph.out_edges[current, next]["cost"]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next)
                heappush(frontier, (priority, next))
                came_from[next] = current

    return cost_so_far[end]


SAMPLE_INPUT = """\
1163751742
1381373672
2136511328
3694931569
7463417111
1319128137
1359912421
3125421639
1293138521
2311944581
"""


@pytest.fixture
def sample_input() -> TextIO:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(("expand_input", "expected"), ((0, 40), (5, 315)))
def test_least_cost_path(sample_input, expand_input, expected):
    risks = read_risks(sample_input)
    graph = build_grid(risks, expand_input)
    result = find_minimum_risk_path(graph, Point(0, 0))
    assert result == expected


if __name__ == "__main__":
    input_path = get_input_path(15, year=2021)
    with Timer() as t:
        main(input_path, t)
