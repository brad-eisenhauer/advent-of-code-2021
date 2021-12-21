""" Advent of Code 2021, Day 21: https://adventofcode.com/2021/day/21 """
from collections import Counter
from dataclasses import dataclass
from functools import cache
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Iterator, TextIO, Iterable, Sequence

import numpy as np
import pylab as p
import pytest

from util import Timer, get_input_path

"""
Part 1:

Because the board is only 10 spaces around, the sums of 3 rolls may be modulo'd by 10,
which yields a descending progression: 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, repeat.

The players will each visit a repeating cycle of spaces over 10 moves (player 1, who
only ever rolls even numbers, actually repeats every 5 turns):

Turn    Player 1                 Player 2
----    ---------------------    ---------------------
  0                     x                        y
  1     +6 =>           x + 6    +5 =>           y + 5
  2     +4 => x + 10 => x        +3 =>           y + 8
  3     +2 =>           x + 2    +1 =>           y + 9
  4     +0 =>           x + 2    +9 => y + 18 => y + 8
  5     +8 => x + 10 => x        +7 => y + 15 => y + 5
  6     +6 =>           x + 6    +5 => y + 10 => y
  7     +4 => x + 10 => x        +3 =>           y + 3
  8     +2 =>           x + 2    +1 =>           y + 4
  9     +0 =>           x + 2    +9 => y + 13 => y + 3
 10     +8 => x + 10 => x        +7 => y + 10 => y
 
Over the course of 10 turns, each player visits a subset of spaces:

We can map out the scores each player achieves on each turn based on their initial
position:

Pos    Player 1                        Player 2
---    ----------------------------    ---------------------------
  1    7 1 3 3 1 7 1 3 3 1     = 30    6 9 10 9 6 1 4 5 4 1  = 55
  2    8 2 4 4 2 8 2 4 4 2     = 40    7 10 1 10 7 2 5 6 5 2 = 55
  3    9 3 5 5 3 9 3 5 5 3     = 50    8 1 2 1 8 3 6 7 6 3   = 45
  4    10 4 6 6 4 10 4 6 6 4   = 60    9 2 3 2 9 4 7 8 7 4   = 55
  5    1 5 7 7 5 1 5 7 7 5     = 50    10 3 4 3 10 5 8 9 8 5 = 65
  6    2 6 8 8 6 2 6 8 8 6     = 60    1 4 5 4 1 6 9 10 9 6  = 55
  7    3 7 9 9 7 3 7 9 9 7     = 70    2 5 6 5 2 7 10 1 10 7 = 55
  8    4 8 10 10 8 4 8 10 10 8 = 80    3 6 7 6 3 8 1 2 1 8   = 45
  9    5 9 1 1 9 5 9 1 1 9     = 50    4 7 8 7 4 9 2 3 2 9   = 55
 10    6 10 2 2 10 6 10 2 2 10 = 60    5 8 9 8 5 10 3 4 3 10 = 65
 
Based on these tables, it is a simple calculation to establish how many turns it would
take each player to reach 1000 points, or how many points each player would have, given
a fixed number of turns.

Part 2:

Three rolls of the Dirac die have seven possible outcomes with varying probabilities:

Total    Occurrences
-----    -----------
    3    1
    4    3
    5    6
    6    7
    7    6
    8    3
    9    1
    
Given a game state, there are seven possible states after the next player's turn, each
representing one or more "universes" in which that new state occurs. In this case,
"state" refers to the combined players' positions and scores. In theory there are
10 * 10 * 21 * 21 ~= 40000 such states, though in practice there will be far fewer.
(Player 1 can't have 20 points, while player 2 has zero, for example.)

Given a game state, we can calculate in how many "next universes" the current player
wins, and recursively count the results for other outcomes.
"""


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        starting_pos = parse_input(fp)
    timer.check("Parse input")

    roll_count, losing_score = calc_rolls_and_losing_score(starting_pos)
    print(losing_score * roll_count)
    timer.check("Part 1, take 1")

    game = initialize_game(starting_pos)
    run_game(game)
    roll_count = game.die.roll_count
    losing_score = min(p.score for p in game.players)
    print(losing_score * roll_count)
    timer.check("Part 1, take 2")

    initial_state = GameState2(
        current_player=Player2(position=starting_pos[0]),
        next_player=Player2(position=starting_pos[1]),
    )
    outcomes = count_outcomes(initial_state)
    print(max(outcomes))
    timer.check("Part 2")


def parse_input(fp: TextIO) -> tuple[int, ...]:
    return tuple(int(line.strip().split()[-1]) for line in fp)


# region Part 1


ROLLS = list((n - 4) % 10 for n in range(10, 0, -1)) * 2
TURN_SCORES = tuple(
    {
        start: [
            (start + total_roll - 1) % 10 + 1
            for total_roll in np.cumsum(ROLLS[player::2])
        ]
        for start in range(10)
    }
    for player in (0, 1)
)


def calc_rolls_and_losing_score(start_positions: tuple[int, ...]) -> tuple[int, int]:
    turns_to_win = tuple(
        count_turns_to_score(player, start, 1000)
        for player, start in enumerate(start_positions)
    )
    turn_count = min(turns_to_win)
    winning_player = min(enumerate(turns_to_win), key=lambda p: p[1])[0]
    losing_player = int(not winning_player)
    losing_score = count_score_after_turns(
        losing_player, start_positions[losing_player], turn_count - losing_player
    )
    roll_count = count_rolls(turn_count, losing_player)
    return roll_count, losing_score


def count_turns_to_score(player: int, starting_space: int, target_score: int) -> int:
    starting_space %= 10
    ten_turn_score = sum(TURN_SCORES[player][starting_space])
    cycles = target_score // ten_turn_score
    score = cycles * ten_turn_score
    result = cycles * 10

    turn_scores = iter(TURN_SCORES[player][starting_space])
    while score < target_score:
        score += next(turn_scores)
        result += 1

    return result


def count_score_after_turns(player: int, starting_space: int, turn_count: int) -> int:
    starting_space %= 10
    ten_turn_score = sum(TURN_SCORES[player][starting_space])
    result = ten_turn_score * (turn_count // 10) + sum(
        TURN_SCORES[player][starting_space][: turn_count % 10]
    )
    return result


def count_rolls(turn_count: int, losing_player: int) -> int:
    total_turn_count = 2 * turn_count - losing_player
    return total_turn_count * 3


# endregion


# region Part 1, take 2
# This approach just simulates the game, brute-force-style. Slower, but easier to verify.


@dataclass
class Player1:
    position: int
    score: int = 0


@dataclass
class DeterministicDie:
    next_roll: int = 1
    roll_count: int = 0

    def get_sum_of_rolls(self, n: int) -> int:
        result = 0
        for _ in range(n):
            result += self.next_roll
            self.roll_count += 1
            self.next_roll = self.next_roll % 100 + 1
        return result


@dataclass
class GameState1:
    players: Sequence[Player1]
    die: DeterministicDie
    next_player: int = 0


def initialize_game(player_positions: Iterable[int]) -> GameState1:
    players = tuple(Player1(position=pos) for pos in player_positions)
    die = DeterministicDie()
    return GameState1(players, die)


def run_game(game: GameState1):
    while max(p.score for p in game.players) < 1000:
        execute_next_turn(game.players[game.next_player], game.die)
        game.next_player = int(not game.next_player)


def execute_next_turn(player: Player1, die: DeterministicDie):
    player.position = (player.position + die.get_sum_of_rolls(3) - 1) % 10 + 1
    player.score += player.position


# endregion


# region Part 2


@dataclass(frozen=True)
class Player2:
    position: int
    # Score counts down to zero as this is more compatible with calculating games of
    # different lengths; the result isn't dependent on the target score for the game,
    # just the remaining points required.
    remaining_score_needed: int = 21


@dataclass(frozen=True)
class GameState2:
    current_player: Player2
    next_player: Player2


TURN_OUTCOMES = Counter(sum(rolls) for rolls in product(*([[1, 2, 3]] * 3)))


@cache
def count_outcomes(state: GameState2) -> tuple[int, int]:
    """returns wins/losses for next_player"""
    result = [0, 0]
    for move, count in TURN_OUTCOMES.items():
        new_pos = (state.current_player.position + move - 1) % 10 + 1
        new_score = state.current_player.remaining_score_needed - new_pos
        if new_score <= 0:
            result[0] += count
        else:
            new_state = GameState2(
                current_player=state.next_player,
                next_player=Player2(position=new_pos, remaining_score_needed=new_score),
            )
            # reverse subsequent results, since we've flipped "current" and "next" players
            other_results = reversed(count_outcomes(new_state))
            result = list(
                current + count * subsequent
                for current, subsequent in zip(result, other_results)
            )

    return tuple(result)


# endregion


SAMPLE_INPUT = """\
Player 1 starting position: 4
Player 2 starting position: 8
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_parse_input(sample_input):
    result = parse_input(sample_input)
    assert result == (4, 8)


def test_count_turns_to_score(sample_input):
    start, _ = parse_input(sample_input)
    result = count_turns_to_score(0, start, 1000)
    assert result == 166


def test_count_score_after_turns(sample_input):
    _, start = parse_input(sample_input)
    result = count_score_after_turns(1, start, 165)
    assert result == 745


@pytest.mark.parametrize(
    ("turn_count", "winning_player", "expected"), ((166, 1, 993), (1, 0, 6))
)
def test_count_rolls(turn_count, winning_player, expected):
    assert count_rolls(turn_count, winning_player) == expected


def test_calc_rolls_and_losing_score(sample_input):
    start_pos = parse_input(sample_input)
    result = calc_rolls_and_losing_score(start_pos)
    assert result == (993, 745)


def test_count_outcomes(sample_input):
    p1_start, p2_start = parse_input(sample_input)
    initial_state = GameState2(Player2(position=p1_start), Player2(position=p2_start))
    result = count_outcomes(initial_state)
    assert result == (444356092776315, 341960390180808)


if __name__ == "__main__":
    input_path = get_input_path(21, year=2021)
    with Timer() as timer:
        main(input_path, timer)
