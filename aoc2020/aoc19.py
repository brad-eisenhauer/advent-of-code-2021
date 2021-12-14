""" Advent of Code 2020, Day 19: https://adventofcode.com/2020/day/19 """
from abc import ABC, abstractmethod
from io import StringIO
from itertools import takewhile
from pathlib import Path
from typing import Iterator, Sequence, Iterable, Union

import pytest

from util import get_input_path, timer


def main(input_path: Path, with_replacement: bool = False):
    with open(input_path) as fp:
        rules = takewhile(lambda line: line != "", (line.strip() for line in fp))
        builder = RuleBuilder(rules)
        messages = [line.strip() for line in fp]

    if with_replacement:
        builder.rules[8] = "42 | 42 8"
        builder.rules[11] = "42 31 | 42 11 31"

    rule_0 = builder.get_rule(0)
    count = sum(1 for message in messages if rule_0.is_match(message))

    print(count)


class Rule(ABC):
    @abstractmethod
    def find_matches(self, message: str) -> Iterator[tuple[str, str]]:
        ...

    def is_match(self, message: str) -> bool:
        return any(remainder == "" for _, remainder in self.find_matches(message))


class LiteralRule(Rule):
    def __init__(self, match: str):
        self.match = match

    def find_matches(self, message: str) -> Iterator[tuple[str, str]]:
        if message.startswith(self.match):
            yield self.match, message[len(self.match) :]


class UnionRule(Rule):
    def __init__(self, rules: Iterable[Rule]):
        self.rules = list(rules)

    def find_matches(self, message: str) -> Iterator[tuple[str, str]]:
        for rule in self.rules:
            yield from rule.find_matches(message)


class SeriesRule(Rule):
    def __init__(self, rules: Iterable[Rule]):
        self.rules = list(rules)

    def find_matches(self, message: str) -> Iterator[tuple[str, str]]:
        def find_matches_rec(
            partial_message: str, rules: Sequence[Rule]
        ) -> Iterator[tuple[str, str]]:
            if len(rules) == 0:
                yield "", partial_message
            else:
                for match, remainder in rules[0].find_matches(partial_message):
                    for sub_match, sub_remainder in find_matches_rec(
                        remainder, rules[1:]
                    ):
                        yield match + sub_match, sub_remainder

        return find_matches_rec(message, self.rules)


class RuleBuilder:
    def __init__(self, rules: Iterable[str]):
        def parse_line(line: str) -> tuple[int, str]:
            n, rule = line.split(":")
            return int(n), rule.strip()

        self.rules: dict[int, Union[str, Rule]] = dict(
            parse_line(line) for line in rules
        )

    def get_rule(self, index: int) -> Rule:
        if isinstance((rule := self.rules[index]), Rule):
            return rule
        self.rules[index] = self.build_rule(rule)
        return self.rules[index]

    def build_rule(self, rule: str) -> Rule:
        if rule.startswith('"'):
            return LiteralRule(rule.strip('"'))
        if "|" in rule:
            return UnionRule(
                self.build_rule(sub_rule.strip()) for sub_rule in rule.split("|")
            )
        return SeriesRule(
            DeferredRule(self, int(sub_rule)) for sub_rule in rule.split()
        )


class DeferredRule(Rule):
    # placeholder that behaves like a rule that hasn't been written yet

    def __init__(self, builder: RuleBuilder, index: int):
        self.builder = builder
        self.index = index

    def find_matches(self, message: str) -> Iterator[tuple[str, str]]:
        return self.builder.get_rule(self.index).find_matches(message)


SAMPLE_INPUT = [
    """\
0: 4 1 5
1: 2 3 | 3 2
2: 4 4 | 5 5
3: 4 5 | 5 4
4: "a"
5: "b"

ababbb
bababa
abbbab
aaabbb
aaaabbb
""",
    """\
42: 9 14 | 10 1
9: 14 27 | 1 26
10: 23 14 | 28 1
1: "a"
11: 42 31
5: 1 14 | 15 1
19: 14 1 | 14 14
12: 24 14 | 19 1
16: 15 1 | 14 14
31: 14 17 | 1 13
6: 14 14 | 1 14
2: 1 24 | 14 4
0: 8 11
13: 14 3 | 1 12
15: 1 | 14
17: 14 2 | 1 7
23: 25 1 | 22 14
28: 16 1
4: 1 1
20: 14 14 | 1 15
3: 5 14 | 16 1
27: 1 6 | 14 18
14: "b"
21: 14 1 | 1 14
25: 1 1 | 1 14
22: 14 14
8: 42
26: 14 22 | 1 20
18: 15 15
7: 14 5 | 1 21
24: 14 1

abbbbbabbbaaaababbaabbbbabababbbabbbbbbabaaaa
bbabbbbaabaabba
babbbbaabbbbbabbbbbbaabaaabaaa
aaabbbbbbaaaabaababaabababbabaaabbababababaaa
bbbbbbbaaaabbbbaaabbabaaa
bbbababbbbaaaaaaaabbababaaababaabab
ababaaaaaabaaab
ababaaaaabbbaba
baabbaaaabbaaaababbaababb
abbbbabbbbaaaababbbbbbaaaababb
aaaaabbaabaaaaababaa
aaaabbaaaabbaaa
aaaabbaabbaaaaaaabbbabbbaaabbaabaaa
babaaabbbaaabaababbaabababaaab
aabbbbbaabbbaaaaaabbbbbababaaaaabbaaabba
""",
]


@pytest.mark.parametrize(
    ("input_index", "with_replacement", "expected"),
    ((0, False, 2), (1, False, 3), (1, True, 12)),
)
def test_count_matching_messages(input_index, with_replacement, expected):
    with StringIO(SAMPLE_INPUT[input_index]) as fp:
        rules = takewhile(lambda line: line != "", (line.strip() for line in fp))
        builder = RuleBuilder(rules)
        messages = list(line.strip() for line in fp)

    if with_replacement:
        builder.rules[8] = "42 | 42 8"
        builder.rules[11] = "42 31 | 42 11 31"

    rule_0 = builder.get_rule(0)
    count = sum(1 for message in messages if rule_0.is_match(message))

    assert count == expected


if __name__ == "__main__":
    input_path = get_input_path(19, year=2020)
    with timer():
        main(input_path, with_replacement=True)
