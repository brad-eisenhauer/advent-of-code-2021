from pathlib import Path
from typing import Callable, Iterable, Tuple, TypeVar

T = TypeVar("T")


def get_input_path(day: int) -> Path:
    return Path(__file__).parent / "resources" / f"input{day:02d}.txt"


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
