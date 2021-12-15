""" Advent of Code 2021, Day 15: https://adventofcode.com/2021/day/15 """
from heapq import heappop, heappush
from io import StringIO
from pathlib import Path
from typing import Iterator, Optional, Sequence, TextIO

import pytest

from util import Timer, get_input_path

Point = tuple[int, int]


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        costs = read_costs(fp)
    timer.check("Read input")

    for expanded in (False, True):
        if expanded:
            costs = expand_costs(costs)
        start = (0, 0)
        result = calc_minimum_cost_to_exit(costs, start)
        print(result)
        timer.check(f"Calc path - expanded={expanded}")


EXPANSION_MULTIPLIER = 5


def read_costs(fp: TextIO) -> Sequence[Sequence[int]]:
    return [list(map(int, line.strip())) for line in fp]


def expand_costs(
    costs: Sequence[Sequence[int]], multiplier: int = EXPANSION_MULTIPLIER
) -> Sequence[Sequence[int]]:
    orig_size = len(costs)
    new_size = orig_size * multiplier
    result = [
        [
            (costs[base_row][base_col] - 1 + row // orig_size + col // orig_size) % 9
            + 1
            for col in range(new_size)
            for base_row in (row % orig_size,)
            for base_col in (col % orig_size,)
        ]
        for row in range(new_size)
    ]
    return result


def calc_minimum_cost_to_exit(
    costs: Sequence[Sequence[int]], start: Point, end: Optional[Point] = None
) -> int:
    """
    A* algorithm based on https://www.redblobgames.com/pathfinding/a-star/introduction.html
    """

    def generate_neighbors(x: int, y: int) -> Iterator[Point]:
        x_range = range(len(costs))
        y_range = range(len(costs[0]))
        if x - 1 in x_range:
            yield x - 1, y
        if x + 1 in x_range:
            yield x + 1, y
        if y - 1 in y_range:
            yield x, y - 1
        if y + 1 in y_range:
            yield x, y + 1

    def get_cost(x: int, y: int) -> int:
        return costs[x][y]

    def heuristic(x: int, y: int) -> int:
        x0, y0 = end
        return abs(x - x0) + abs(y - y0)

    if end is None:
        end = (len(costs) - 1, len(costs[0]) - 1)

    frontier = []
    heappush(frontier, (0, start))
    accumulated_cost = {start: 0}

    while len(frontier) > 0:
        _, current = heappop(frontier)
        if current == end:
            break

        for next in generate_neighbors(*current):
            new_cost = accumulated_cost[current] + get_cost(*next)
            if next not in accumulated_cost or new_cost < accumulated_cost[next]:
                accumulated_cost[next] = new_cost
                priority = new_cost + heuristic(*current)
                heappush(frontier, (priority, next))

    return accumulated_cost[end]


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


@pytest.mark.parametrize(("expand_input", "expected"), ((False, 40), (True, 315)))
def test_least_cost_path(sample_input, expand_input, expected):
    costs = read_costs(sample_input)
    if expand_input:
        costs = expand_costs(costs)
    result = calc_minimum_cost_to_exit(costs, (0, 0))
    assert result == expected


if __name__ == "__main__":
    input_path = get_input_path(15, year=2021)
    with Timer() as t:
        main(input_path, t)
