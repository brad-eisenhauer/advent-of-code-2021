import time
from contextlib import contextmanager
from datetime import date
from itertools import islice, tee
from os import getenv
from pathlib import Path
from typing import (
    Callable,
    ContextManager,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Iterator,
)

import requests
from dotenv import load_dotenv

load_dotenv()
T = TypeVar("T")


def get_input_path(day: int, year: int) -> Path:
    resources_path = Path(__file__).parent / "resources" / str(year)
    input_path = resources_path / f"input{day:02d}.txt"

    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        download_input(input_path, day, year or date.today().year)

    return input_path


def download_input(download_path: Path, day: int, year: int):
    download_url = f"https://adventofcode.com/{year}/day/{day}/input"
    response = requests.get(download_url, cookies={"session": getenv("AOC_SESSION")})
    response.raise_for_status()
    with open(download_path, "w") as fp:
        fp.write(response.content.decode())


def partition(
    predicate: Callable[[T], bool], items: Iterable[T]
) -> tuple[Sequence[T], Sequence[T]]:
    """
    Split items into two lists, based on some predicate condition

    Parameters
    ----------
    predicate
        Boolean function by which to categorize items
    items
        Items to be categorized

    Returns
    -------
    Sequences of failing and passing items, respectively
    """
    result = [], []
    for item in items:
        result[int(predicate(item))].append(item)
    return result


def greatest_common_divisor(a: int, b: int) -> int:
    if b == 0:
        return abs(a)
    return greatest_common_divisor(b, a % b)


@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {(end - start) * 1000} ms")


def make_sequence(items: Iterable[T]) -> Sequence[T]:
    if isinstance(items, Sequence):
        return items
    return tuple(items)


def create_windows(items: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    iterators = tee(items, n)
    offset_iterators = (
        islice(iterator, offset, None) for offset, iterator in enumerate(iterators)
    )
    return zip(*offset_iterators)


class Timer(ContextManager):
    def __init__(self):
        self.start: float = 0.0
        self.last_check: Optional[float] = None
        self.check_index = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.check("Elapsed time")

    def check(self, label: Optional[str] = None):
        check_time = time.time()
        self.check_index += 1
        message = (
            f"[{self.check_index}] {label or 'Time check'}: "
            f"{self.get_formatted_time(check_time)}"
        )
        message += self.get_last_check_msg(check_time)
        print(message)
        self.last_check = check_time

    def get_formatted_time(self, end: float, start: Optional[float] = None) -> str:
        start = start or self.start
        return f"{(end - start) * 1000:0.3f} ms"

    def get_last_check_msg(self, check_time: float) -> str:
        if self.last_check is None:
            return ""
        return (
            f" ({self.get_formatted_time(check_time, self.last_check)} "
            "since last check)"
        )
