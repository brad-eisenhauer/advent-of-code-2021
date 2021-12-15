""" Advent of Code 2021, Day 12: https://adventofcode.com/2021/day/12 """
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TextIO

import networkx as nx
import pytest

from util import Timer, get_input_path

Cave = str


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        graph = read_graph(fp)
    timer.check("Read input")

    # part 1
    result = count_paths_dfs(graph)
    print(result)
    timer.check("Part 1")

    # part 2
    result = count_paths_dfs(graph, True)
    print(result)
    timer.check("Part 2")


def read_graph(fp: TextIO) -> nx.Graph:
    graph = nx.Graph()
    for line in fp:
        left, right = line.strip().split("-")
        graph.add_edge(left, right)
    return graph


@dataclass(frozen=True)
class PathState:
    last_cave: Cave
    caves_visited: frozenset[Cave]
    has_revisited_small_cave: bool


INITIAL_PATH_STATE = PathState("start", frozenset(), False)


def count_paths_bfs(graph: nx.Graph, allow_revisit_to_small_cave: bool = False) -> int:
    queue = [INITIAL_PATH_STATE]
    path_count = 0

    while len(queue) > 0:
        state = queue.pop(0)

        caves_visited = state.caves_visited | {state.last_cave}
        for next_cave in graph.neighbors(state.last_cave):
            if next_cave == "start":
                continue
            if next_cave == "end":
                path_count += 1
                continue

            if is_big(next_cave) or next_cave not in state.caves_visited:
                queue.append(
                    PathState(next_cave, caves_visited, state.has_revisited_small_cave)
                )
            # small, previously-visited cave: check whether we're allowed to revisit
            elif allow_revisit_to_small_cave and not state.has_revisited_small_cave:
                queue.append(PathState(next_cave, caves_visited, True))

    return path_count


def count_paths_dfs(graph: nx.Graph, allow_revisit_to_small_cave: bool = False) -> int:
    def count_paths_rec(state: PathState) -> int:
        path_count = 0
        caves_visited = state.caves_visited | {state.last_cave}

        for next_cave in graph.neighbors(state.last_cave):
            if next_cave == "start":
                continue
            if next_cave == "end":
                path_count += 1
                continue

            if is_big(next_cave) or next_cave not in state.caves_visited:
                path_count += count_paths_rec(
                    PathState(next_cave, caves_visited, state.has_revisited_small_cave)
                )
            # small, previously-visited cave: check if we're allowed to revisit
            elif allow_revisit_to_small_cave and not state.has_revisited_small_cave:
                path_count += count_paths_rec(PathState(next_cave, caves_visited, True))

        return path_count

    return count_paths_rec(INITIAL_PATH_STATE)


def is_big(cave: Cave) -> bool:
    return cave.isupper()


SAMPLE_INPUT = [
    """start-A
start-b
A-c
A-b
b-d
A-end
b-end
""",
    """dc-end
HN-start
start-kj
dc-start
dc-HN
LN-dc
HN-end
kj-sa
kj-HN
kj-dc
""",
    """fs-end
he-DX
fs-he
start-DX
pj-DX
end-zg
zg-sl
zg-pj
pj-he
RW-he
fs-DX
pj-RW
zg-RW
start-pj
he-WI
zg-he
pj-fs
start-RW
""",
]


@pytest.mark.parametrize(
    ("sample_index", "revisit_small_cave", "expected", "count_paths"),
    (
        (0, False, 10, count_paths_bfs),
        (1, False, 19, count_paths_bfs),
        (2, False, 226, count_paths_bfs),
        (0, True, 36, count_paths_bfs),
        (1, True, 103, count_paths_bfs),
        (2, True, 3509, count_paths_bfs),
        (0, False, 10, count_paths_dfs),
        (1, False, 19, count_paths_dfs),
        (2, False, 226, count_paths_dfs),
        (0, True, 36, count_paths_dfs),
        (1, True, 103, count_paths_dfs),
        (2, True, 3509, count_paths_dfs),
    ),
)
def test_count_paths(sample_index: int, revisit_small_cave, expected, count_paths):
    with StringIO(SAMPLE_INPUT[sample_index]) as sample_input:
        graph = read_graph(sample_input)

    count = count_paths(graph, revisit_small_cave)
    assert count == expected


if __name__ == "__main__":
    input_path = get_input_path(12)
    with Timer() as t:
        main(input_path, t)
