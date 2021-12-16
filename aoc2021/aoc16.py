""" Advent of Code 2021, Day 16: https://adventofcode.com/2021/day/16 """
import operator
from abc import ABC, abstractmethod
from functools import reduce
from itertools import islice
from pathlib import Path
from typing import Iterator, Sequence

import pytest

from util import Timer, get_input_path

bit = int


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        hex = fp.readline()
    timer.check("Read input")

    bits = generate_bits(hex)
    packet = parse_packet(bits)
    timer.check("Parse packets")

    print(packet.sum_of_versions())
    timer.check("Part 1")

    print(packet.evaluate())
    timer.check("Part 2")


class Packet(ABC):
    def __init__(self, packet_type: int, version: int):
        self.type = packet_type
        self.version = version

    @abstractmethod
    def sum_of_versions(self):
        ...

    @abstractmethod
    def evaluate(self) -> int:
        ...


class Literal(Packet):
    def __init__(self, packet_type: int, version: int, value: int):
        super().__init__(packet_type, version)
        self.value = value

    def sum_of_versions(self):
        return self.version

    def evaluate(self) -> int:
        return self.value


class Operation(Packet):
    OPERATORS = {
        0: operator.add,
        1: operator.mul,
        2: min,
        3: max,
        5: operator.gt,
        6: operator.lt,
        7: operator.eq,
    }
    BINARY_OPERATIONS = {5, 6, 7}

    def __init__(self, packet_type: int, version: int, operands: Sequence[Packet]):
        if packet_type in self.BINARY_OPERATIONS and len(operands) != 2:
            raise ValueError(
                f"Binary operation ({packet_type}) requires exactly 2 operands; "
                f"received {len(operands)}."
            )
        super().__init__(packet_type, version)
        self.operands = operands

    def sum_of_versions(self):
        return self.version + sum(p.sum_of_versions() for p in self.operands)

    def evaluate(self) -> int:
        op = self.OPERATORS[self.type]
        return int(reduce(op, (p.evaluate() for p in self.operands)))


def generate_bits(hex: str) -> Iterator[bit]:
    for c in hex:
        value = int(c, 16)
        for offset in range(3, -1, -1):
            yield 1 & (value >> offset)


def parse_packet(bits: Iterator[bit]) -> Packet:
    version = value_of_bits(bits, 3)
    packet_type = value_of_bits(bits, 3)

    if packet_type == 4:
        value = read_literal_value(bits)
        return Literal(packet_type, version, value)

    # Operator packet
    if value_of_bits(bits, 1):
        packet_count = value_of_bits(bits, 11)
        sub_packets = [parse_packet(bits) for _ in range(packet_count)]
    else:
        sub_packet_length = value_of_bits(bits, 15)
        sub_packet_bits = islice(bits, sub_packet_length)
        sub_packets = list(parse_all_packets(sub_packet_bits))
    return Operation(packet_type, version, sub_packets)


def value_of_bits(bits: Iterator[bit], length: int) -> int:
    result = 0
    for _ in range(length):
        result <<= 1
        result += next(bits)
    return result


def read_literal_value(bits: Iterator[bit]) -> int:
    value = 0
    keep_reading = True
    while keep_reading:
        keep_reading = value_of_bits(bits, 1)
        value <<= 4
        value += value_of_bits(bits, 4)
    return value


def parse_all_packets(bits: Iterator[bit]) -> Iterator[Packet]:
    try:
        while packet := parse_packet(bits):
            yield packet
    except StopIteration:
        ...


@pytest.mark.parametrize(
    ("hex", "expected"),
    (
        ("8A004A801A8002F478", 16),
        ("620080001611562C8802118E34", 12),
        ("C0015000016115A2E0802F182340", 23),
        ("A0016C880162017C3686B18A3D4780", 31),
    ),
)
def test_sum_of_versions(hex, expected):
    bits = generate_bits(hex)
    packet = parse_packet(bits)
    assert packet.sum_of_versions() == expected
    assert sum(bits) == 0  # Have we consumed all the meaningful bits?


@pytest.mark.parametrize(
    ("hex", "expected"),
    (
        ("C200B40A82", 3),
        ("04005AC33890", 54),
        ("880086C3E88112", 7),
        ("CE00C43D881120", 9),
        ("D8005AC2A8F0", 1),
        ("F600BC2D8F", 0),
        ("9C005AC2F8F0", 0),
        ("9C0141080250320F1802104A08", 1),
        ("D24A", 42),  # literal 42 (multiple quads)
        ("A600AC3587", 42),  # product 6 * 7
    ),
)
def test_evaluate(hex, expected):
    bits = generate_bits(hex)
    packet = parse_packet(bits)
    assert packet.evaluate() == expected
    assert sum(bits) == 0  # Have we consumed all the meaningful bits?


def test_generate_bits():
    assert list(generate_bits("50A")) == [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]


def test_value_of_bits():
    bits = iter((1, 0, 1, 0, 0, 1, 0, 0))
    assert value_of_bits(bits, 6) == 41
    assert sum(1 for b in bits if b == 0) == 2  # 2 zeros left over


if __name__ == "__main__":
    input_path = get_input_path(16, year=2021)
    with Timer() as t:
        main(input_path, t)
