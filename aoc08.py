""" Advent of Code 2021, Day 08: https://adventofcode.com/2021/day/8"""
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Iterator, TextIO, Tuple

from util import get_input_path, timer


def main(input_path: Path):
    with open(input_path) as fp:
        result = sum_displays(fp)

    print(result)


def count_easy_digits(fp: TextIO) -> int:
    target_values = {"1", "4", "7", "8"}
    count = 0
    for patterns, display in map(parse_line, fp):
        count += sum(
            1 for char in decode_display(patterns, display) if char in target_values
        )
    return count


def sum_displays(fp: TextIO) -> int:
    return sum(
        int(decode_display(patterns, display))
        for patterns, display in map(parse_line, fp)
    )


def parse_line(line: str) -> Tuple[Iterator[str], Iterator[str]]:
    patterns, display = line.split("|")
    return (p for p in patterns.strip().split()), (d for d in display.strip().split())


def decode_display(patterns: Iterable[str], display: Iterable[str]) -> str:
    decode = build_decoder(patterns)
    return "".join(decode(digit) for digit in display)


def build_decoder(patterns: Iterable[str]) -> Callable[[str], str]:
    """
    - Digits 1, 4, 7, and 8 can be identified from length alone.
    - Remaining digits map to lengths as follows:
        Length  Digits
        ------  ------
        6       0 6 9
        5       2 5 3
    - Segments appear with the following frequencies:
        Segment   Count
        --------  -----
        top       8
        middle    7
        bottom    7
        up-left   6
        up-right  8
        lo-left   4
        lo-right  9
    - upper-left, lower-left, and lower-right segments can be identified by the number
        of their appearances in the pattern data.

    - 9 can be distinguished from 0 and 6, as it does not contain the lower-left segment.
    - 6 and be distinguished from 0 and 9 by comparison with 1; 6 is the only digit
        that does not contain both segments from 1.
    - 2 can be distinguished from 3 and 5, as it contains the lower-left segment.
    - 3 can be distinguished from 2 and 5 by comparison with 1; 3 is the only digit
        that contains both segments.
    """
    # Get pattern representing "1" and segment appearance counts
    one_pattern = None
    segment_appearance_counts = defaultdict(int)
    for pattern in patterns:
        if one_pattern is None and len(pattern) == 2:
            one_pattern = pattern
        for segment in pattern:
            segment_appearance_counts[segment] += 1
    lower_left_segment = next(
        segment for segment, count in segment_appearance_counts.items() if count == 4
    )
    digits_by_segment_count = {2: "1", 4: "4", 3: "7", 7: "8"}

    def decode(display: str) -> str:
        segment_count = len(display)
        if segment_count in digits_by_segment_count:
            return digits_by_segment_count[segment_count]
        if segment_count == 6:  # 0, 6, 9
            if lower_left_segment not in display:
                return "9"
            if any(segment not in display for segment in one_pattern):
                return "6"
            return "0"
        if segment_count == 5:  # 2, 3, 5
            if lower_left_segment in display:
                return "2"
            if all(segment in display for segment in one_pattern):
                return "3"
            return "5"
        raise ValueError(f"Invalid pattern: '{display}'")

    return decode


TEST_INPUT = """be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce
"""


def test_count_easy_numbers():
    with StringIO(TEST_INPUT) as fp:
        count = count_easy_digits(fp)

    assert count == 26


def test_sum_displays():
    with StringIO(TEST_INPUT) as fp:
        result = sum_displays(fp)

    assert result == 61229


if __name__ == "__main__":
    with timer():
        main(get_input_path(8))
