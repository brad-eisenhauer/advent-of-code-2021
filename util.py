import time
from contextlib import contextmanager
from datetime import date
from os import getenv
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import requests
from dotenv import load_dotenv

load_dotenv()
T = TypeVar("T")


def get_input_path(day: int, year: Optional[int] = None) -> Path:
    resources_path = Path(__file__).parent / "resources"
    if year is not None:
        resources_path /= str(year)
    input_path = resources_path / f"input{day:02d}.txt"

    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        download_input(input_path, day, year)

    return input_path


def download_input(download_path: Path, day: int, year: Optional[int] = None):
    if year is None:
        year = date.today().year
    download_url = f"https://adventofcode.com/{year}/day/{day}/input"
    with open(download_path, "w") as fp:
        response = requests.get(download_url, cookies={"session": getenv("session")})
        response.raise_for_status()
        fp.write(response.content.decode())


def partition(
    predicate: Callable[[T], bool], items: Iterable[T]
) -> Tuple[list[T], list[T]]:
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
    Lists of failing and passing items, respectively
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
