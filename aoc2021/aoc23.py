""" Advent of Code 2021, Day 23: https://adventofcode.com/2021/day/23 """
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Iterator, Optional, TextIO

import networkx as nx
import pytest

from util import Timer, get_input_path


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        initial_state = read_state(fp)
    timer.check("Read input")

    cost_to_solve = find_minimal_cost_to_solution(initial_state)
    print(cost_to_solve)
    timer.check("Part 1")

    with open(input_path) as fp:
        initial_state = read_state(extend_state(fp))
    timer.check("Read input")

    cost_to_solve = find_minimal_cost_to_solution(initial_state, build_board(extended=True))
    print(cost_to_solve)
    timer.check("Part 2")


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def in_hallway(self):
        return self.y == 0

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)


def build_board(extended: bool = False) -> nx.Graph:
    g = nx.Graph()
    # hallway is 11 units long
    g.add_edges_from(
        (Position(x, 0), Position(x + 1, 0)) for x in range(10)
    )
    # add each room
    for room_x in range(2, 10, 2):
        g.add_edges_from((
            (Position(room_x, 0), Position(room_x, 1)),
            (Position(room_x, 1), Position(room_x, 2)),
        ))
        if extended:
            g.add_edges_from((
                (Position(room_x, 2), Position(room_x, 3)),
                (Position(room_x, 3), Position(room_x, 4)),
            ))
    return g


BOARD = build_board()


class AmphipodType(Enum):
    def __new__(cls, *args, **kwargs) -> AmphipodType:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, movement_cost: int, destination_x: int):
        self.movement_cost = movement_cost
        self.destination_x = destination_x

    Amber = "A", 1, 2
    Bronze = "B", 10, 4
    Copper = "C", 100, 6
    Desert = "D", 1000, 8


@dataclass(frozen=True)
class AmphipodState:
    type: AmphipodType
    position: Position
    at_rest: bool = False

    def at_destination(self):
        return (
            not self.position.in_hallway()
            and self.position.x == self.type.destination_x
        )

    def __lt__(self, other):
        return self.position < other.position


@dataclass(frozen=True)
class BoardState:
    amphipods: frozenset[AmphipodState]
    last_moved: Optional[AmphipodState] = None

    def is_goal_state(self):
        return all(a.at_destination() for a in self.amphipods)

    @cache
    def get_occupied_positions(self) -> frozenset[Position]:
        return frozenset(pod.position for pod in self.amphipods)

    def is_position_occupied(self, position: Position) -> bool:
        return position in self.get_occupied_positions()

    def __lt__(self, other: BoardState):
        return tuple(self.amphipods) < tuple(other.amphipods)


def read_state(fp: TextIO) -> BoardState:
    amphipods = []
    _ = fp.readline()  # discard first line
    for y, line in enumerate(fp):
        for x, c in enumerate(line[1:]):
            if c in ("A", "B", "C", "D"):
                amphipods.append(
                    AmphipodState(type=AmphipodType(c), position=Position(x, y))
                )

    return BoardState(amphipods=frozenset(amphipods))


def extend_state(fp: TextIO) -> StringIO:
    result_lines = []
    for i, line in enumerate(fp):
        if i == 3:
            result_lines.extend(["  #D#C#B#A#\n", "  #D#B#A#C#\n"])
        result_lines.append(line)
    result = StringIO("".join(result_lines))
    return result


def find_minimal_cost_to_solution(initial_state: BoardState, board: nx.Graph = BOARD) -> int:
    """ Dijkstra path-finding algorithm """
    frontier: list[tuple[float, BoardState]] = []
    heapq.heappush(frontier, (0, initial_state))

    came_from = {}
    cost_so_far = {}
    came_from[initial_state] = None
    cost_so_far[initial_state] = 0

    while len(frontier) > 0:
        _, current = heapq.heappop(frontier)

        if current.is_goal_state():
            return cost_so_far[current]

        for next_state, next_cost in generate_next_states(current, board):
            new_cost = cost_so_far[current] + next_cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current


def generate_next_states(state: BoardState, board: nx.Graph) -> Iterator[tuple[BoardState, int]]:
    """ Generate legal successor states and their costs """
    room_depth = len(state.amphipods) // 4

    def generate_legal_moves(pod: AmphipodState) -> Iterator[tuple[AmphipodState, int]]:
        # If a pod is in its home room, and no pods of a different type are below it,
        # move to the lowest available position.
        if (
            pod.position.x == pod.type.destination_x
            and not pod.position.in_hallway()
            and all(p.type is p for p in state.amphipods if p.position.x == pod.position.x and p.position.y > pod.position.y)
        ):
            dest_y = pod.position.y + 1
            while dest_y <= room_depth and not state.is_position_occupied(replace(pod.position, y=dest_y)):
                dest_y += 1
            dest_y -= 1
            if dest_y > pod.position.y:
                yield replace(pod, position=replace(pod.position, y=dest_y), at_rest=True), dest_y - pod.position.y
            return

        # If a pod is stopped in the hallway, it can only move if it can reach its
        # destination, and then it must move toward its destination or into its room.
        if pod.position.in_hallway() and can_reach_destination(pod):
            dest_y = 1
            while dest_y <= room_depth and not state.is_position_occupied(Position(pod.type.destination_x, dest_y)):
                dest_y += 1
            dest_y -= 1
            yield replace(pod, position=Position(pod.type.destination_x, dest_y), at_rest=True), abs(pod.position.x - pod.type.destination_x) + dest_y
            return
        if pod.position.in_hallway() and pod is not state.last_moved:
            return

        # If a pod is in a room and all positions above it are clear, move out of the room.
        if (
            not pod.position.in_hallway()
            and all(
                not state.is_position_occupied(replace(pod.position, y=y))
                for y in range(0, pod.position.y)
            )
        ):
            yield replace(pod, position=replace(pod.position, y=0)), pod.position.y
            return

        for next_pos in board.neighbors(pod.position):
            # All legal moves out of the hallway have been covered above.
            if pod.position.in_hallway() and not next_pos.in_hallway():
                continue
            if state.is_position_occupied(next_pos):
                continue
            yield replace(pod, position=next_pos), 1

    def can_reach_destination(pod: AmphipodState) -> bool:
        here = pod.position.x
        there = pod.type.destination_x
        clear_range = range(there, here, sign(here - there))

        if any(state.is_position_occupied(Position(x, 0)) for x in clear_range):
            return False
        if state.is_position_occupied(Position(there, 1)):
            return False
        if any(p.type is not pod.type and p.position.x == there for p in state.amphipods):
            return False
        return True

    # If the last amphipod to move is just outside a room, it must continue moving
    if (
        state.last_moved is not None
        and state.last_moved.position.in_hallway()
        and state.last_moved.position.x in (2, 4, 6, 8)
    ):
        for next_position, move_count in generate_legal_moves(state.last_moved):
            yield (
                move_pod(state, state.last_moved, next_position),
                state.last_moved.type.movement_cost * move_count,
            )
        return

    # Otherwise, loop through legal successor states
    for pod in state.amphipods:
        if not pod.at_rest:
            for next_position, move_count in generate_legal_moves(pod):
                yield move_pod(state, pod, next_position), pod.type.movement_cost * move_count


def move_pod(state: BoardState, pod: AmphipodState, new_pod: AmphipodState) -> BoardState:
    amphipods = frozenset(
        new_pod if a is pod else a for a in state.amphipods
    )
    return BoardState(amphipods, new_pod)


def heuristic(state: BoardState) -> int:
    """Minimum cost for all pods to move directly 'home', without impediment"""
    result = 0
    room_depth = len(state.amphipods) // 4
    count_need_to_go_home_by_type = defaultdict(int)
    for pod in state.amphipods:
        if pod.position.in_hallway():
            result += pod.type.movement_cost * abs(pod.position.x - pod.type.destination_x)
            count_need_to_go_home_by_type[pod.type] += 1
        # If pod is in the wrong room, it needs to move to the hallway, over, and back
        # down into its room.
        elif pod.position.x != pod.type.destination_x:
            result += pod.type.movement_cost * (pod.position.y + abs(pod.position.x - pod.type.destination_x))
            count_need_to_go_home_by_type[pod.type] += 1
        # If there are pods of a different type "below" this pod, it will have to move
        # to the hallway, out of the way, and back into the room.
        # But this is more expensive to compute than it's worth.
        # elif any(p.position.x == pod.position.x and p.position.y > pod.position.y for p in state.amphipods):
        #     result += pod.type.movement_cost * (pod.position.y + 2)
        #     count_need_to_go_home_by_type[pod.type] += 1
        # Otherwise, pod just needs to move to the bottom of its room
        else:
            result += pod.type.movement_cost * (room_depth - pod.position.y)

    for type, count in count_need_to_go_home_by_type.items():
        result += type.movement_cost * count * (count + 1) // 2

    return result


def sign(n: int) -> int:
    if n < 0:
        return -1
    return 1


SAMPLE_INPUT = """\
#############
#...........#
###B#C#B#D###
  #A#D#C#A#
  #########
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


@pytest.mark.parametrize(
    ("name", "movement_cost"), (("A", 1), ("B", 10), ("C", 100), ("D", 1000))
)
def test_amphipod_types(name, movement_cost):
    assert AmphipodType(name).movement_cost == movement_cost


INITIAL_STATE = BoardState(
        amphipods=frozenset((
            AmphipodState(type=AmphipodType.Bronze, position=Position(2, 1)),
            AmphipodState(type=AmphipodType.Copper, position=Position(4, 1)),
            AmphipodState(type=AmphipodType.Bronze, position=Position(6, 1)),
            AmphipodState(type=AmphipodType.Desert, position=Position(8, 1)),
            AmphipodState(type=AmphipodType.Amber, position=Position(2, 2)),
            AmphipodState(type=AmphipodType.Desert, position=Position(4, 2)),
            AmphipodState(type=AmphipodType.Copper, position=Position(6, 2)),
            AmphipodState(type=AmphipodType.Amber, position=Position(8, 2)),
        ))
    )
SOLUTION_MOVES = (
    (Position(6, 1), Position(6, 0), False),
    (Position(6, 0), Position(5, 0), False),
    (Position(5, 0), Position(4, 0), False),
    (Position(4, 0), Position(3, 0), False),
    (Position(4, 1), Position(4, 0), False),
    (Position(4, 0), Position(6, 1), True),
    (Position(4, 2), Position(4, 0), False),
    (Position(4, 0), Position(5, 0), False),
    (Position(3, 0), Position(4, 2), True),
    (Position(2, 1), Position(2, 0), False),
    (Position(2, 0), Position(4, 1), True),
    (Position(8, 1), Position(8, 0), False),
    (Position(8, 0), Position(7, 0), False),
    (Position(8, 2), Position(8, 0), False),
    (Position(8, 0), Position(9, 0), False),
    (Position(7, 0), Position(8, 2), True),
    (Position(5, 0), Position(8, 1), True),
    (Position(9, 0), Position(2, 1), True),
)


def generate_solution_states() -> Iterator[BoardState]:
    state = INITIAL_STATE
    yield state
    for prev_pos, new_pos, at_rest in SOLUTION_MOVES:
        pod = next(p for p in state.amphipods if p.position == prev_pos)
        state = move_pod(state, pod, replace(pod, position=new_pos, at_rest=at_rest))
        yield state


SOLUTION_STATES = tuple(generate_solution_states())


def test_read_state(sample_input):
    state = read_state(sample_input)
    expected = INITIAL_STATE
    assert state == expected


def test_move_pod():
    state = INITIAL_STATE
    for prev_pos, new_pos, at_rest in SOLUTION_MOVES:
        assert not state.is_position_occupied(new_pos)
        pod = next(p for p in state.amphipods if p.position == prev_pos)
        state = move_pod(state, pod, replace(pod, position=new_pos, at_rest=at_rest))
    assert state.is_goal_state()


def test_generate_board_states():
    state = INITIAL_STATE
    for prev_pos, next_pos, at_rest in SOLUTION_MOVES:
        pod = next(p for p in state.amphipods if p.position == prev_pos)
        expected_state = move_pod(state, pod, replace(pod, position=next_pos, at_rest=at_rest))
        next_states = set(s for s, _ in generate_next_states(state, BOARD))
        assert expected_state in next_states
        state = expected_state


@pytest.mark.parametrize(
    ("state_index", "expected_cost"),
    ((-2, 8), (-4, 7008), (-8, 9011), (-10, 9051), (6, 12081), (0, 12521))
)
def test_find_minimal_cost_solution(state_index, expected_cost):
    state = SOLUTION_STATES[state_index]
    cost = find_minimal_cost_to_solution(state)
    assert cost == expected_cost


if __name__ == "__main__":
    input_path = get_input_path(23, year=2021)
    with Timer() as timer:
        main(input_path, timer)
