"""Count increasing windows"""

from itertools import islice, tee
from pathlib import Path
from typing import Iterable, Iterator, Tuple, TypeVar

INPUT_FILE = Path(__file__).parent / "resources" / "input01.txt"
WINDOW_SIZE = 1

T = TypeVar("T")


def main():
    with open(INPUT_FILE) as fp:
        measurements = (int(line) for line in fp.readlines())

    windows = create_windows(measurements, WINDOW_SIZE)
    window_sums = (sum(window) for window in windows)
    result = sum(
        1
        for predecessor, successor in create_windows(window_sums, 2)
        if successor > predecessor
    )

    print(result)


def create_windows(items: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    iterators = tee(items, n)
    offset_iterators = (
        islice(iterator, offset, None) for offset, iterator in enumerate(iterators)
    )
    return zip(*offset_iterators)


def main2():
    with open(INPUT_FILE) as fp:
        measurements = (int(line) for line in fp.readlines())

        window = list(islice(measurements, WINDOW_SIZE))
        prev_window_sum = sum(window)
        result = 0
        for measurement in measurements:
            window_sum = prev_window_sum - window.pop(0) + measurement
            window.append(measurement)
            if window_sum > prev_window_sum:
                result += 1
            prev_window_sum = window_sum

    print(result)


if __name__ == "__main__":
    main()
