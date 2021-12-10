"""AoC 2021, Day 04: https://adventofcode.com/2021/day/4"""
from io import StringIO
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO, Tuple, Union

from util import get_input_path, partition, timer


def main(input_path: Union[Path, str]):
    with open(input_path) as fp:
        boards, draws = read_game_input(fp)

    scores = run_game(boards, draws)

    print(f"Winning/losing scores: {scores}")


class Board:
    SIZE = 5

    def __init__(self, lines: Iterable[str]):
        self.numbers = tuple(int(value) for line in lines for value in line.split())
        if len(self.numbers) != len(self):
            raise ValueError(
                f"Incorrect number of values given. Expected {len(self)}, received {len(self.numbers)}."
            )
        self.markers = [False] * len(self)

    def mark(self, value: int):
        try:
            index = self.numbers.index(value)
            self.markers[index] = True
        except ValueError:
            ...

    @property
    def has_won(self) -> bool:
        # Check rows
        if any(
            all(self.markers[i : i + self.SIZE]) for i in range(0, len(self), self.SIZE)
        ):
            return True
        # Check columns
        if any(all(self.markers[i :: self.SIZE]) for i in range(self.SIZE)):
            return True
        return False

    @property
    def score(self) -> int:
        return sum(
            number for number, marked in zip(self.numbers, self.markers) if not marked
        )

    def __len__(self):
        return self.SIZE ** 2


def read_game_input(fp: TextIO) -> Tuple[Sequence[Board], Iterable[int]]:
    draws = (int(value) for value in fp.readline().strip().split(","))
    boards = []
    while (board := read_board(fp)) is not None:
        boards.append(board)
    return boards, draws


def read_board(fp: TextIO) -> Optional[Board]:
    try:
        fp.readline()  # throw away blank line
        return Board(fp.readline() for _ in range(Board.SIZE))
    except (IOError, ValueError):
        return None


def run_game(boards: Sequence[Board], draws: Iterable[int]) -> Tuple[int, int]:
    first_winning_score = 0
    last_winning_score = 0

    draws = iter(draws)
    for number in islice(draws, Board.SIZE - 1):
        mark_boards(boards, number)

    for number in draws:
        mark_boards(boards, number)
        boards, winning_boards = partition(lambda board: board.has_won, boards)
        if winning_boards:
            first_winning_score = first_winning_score or max(
                board.score * number for board in winning_boards
            )
        if len(boards) == 0:
            last_winning_score = min(board.score * number for board in winning_boards)
            break

    return first_winning_score, last_winning_score


def mark_boards(boards: Iterable[Board], value: int):
    for board in boards:
        board.mark(value)


TEST_INPUT = """7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1

22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7
"""


def test_game():
    with StringIO(TEST_INPUT) as fp:
        boards, draws = read_game_input(fp)

    scores = run_game(boards, draws)

    assert (4512, 1924) == scores


if __name__ == "__main__":
    input_path = get_input_path(4)
    with timer():
        main(input_path)
