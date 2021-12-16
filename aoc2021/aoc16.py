""" Advent of Code 2021, Day 16: https://adventofcode.com/2021/day/16 """
import operator
from functools import reduce
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import pytest

from util import Timer, get_input_path


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


class Packet:
    def __init__(self, ptype: int, version: int):
        self.ptype = ptype
        self.version = version

    def sum_of_versions(self):
        return self.version

    def evaluate(self) -> int:
        ...


class LiteralPacket(Packet):
    def __init__(self, ptype: int, version: int, value: int):
        super().__init__(ptype, version)
        self.value = value

    def sum_of_versions(self):
        return self.version

    def evaluate(self) -> int:
        return self.value


class OperatorPacket(Packet):
    def __init__(self, ptype: int, version: int, sub_packets: Sequence[Packet]):
        super().__init__(ptype, version)
        self.sub_packets = sub_packets

    def sum_of_versions(self):
        return self.version + sum(p.sum_of_versions() for p in self.sub_packets)

    def evaluate(self) -> int:
        if self.ptype == 0:
            return sum(p.evaluate() for p in self.sub_packets)
        if self.ptype == 1:
            return reduce(operator.mul, (p.evaluate() for p in self.sub_packets))
        if self.ptype == 2:
            return min(p.evaluate() for p in self.sub_packets)
        if self.ptype == 3:
            return max(p.evaluate() for p in self.sub_packets)
        if self.ptype == 5:
            left, right = self.sub_packets
            return int(left.evaluate() > right.evaluate())
        if self.ptype == 6:
            left, right = self.sub_packets
            return int(left.evaluate() < right.evaluate())
        if self.ptype == 7:
            left, right = self.sub_packets
            return int(left.evaluate() == right.evaluate())
        raise ValueError(f"Unrecognized packet type: {self.ptype}")


def generate_bits(hex: str) -> Iterator[str]:
    for c in hex:
        yield from f"{int(c, 16):04b}"


def parse_packet(bits: Iterable[str]) -> Optional[Packet]:
    try:
        version = value_of_bits(bits, 3)
    except ValueError:
        return None

    ptype = value_of_bits(bits, 3)

    if ptype == 4:
        value = 0
        keep_reading = True
        while keep_reading:
            keep_reading = value_of_bits(bits, 1)
            value *= 16
            value += value_of_bits(bits, 4)
        return LiteralPacket(ptype, version, value)

    # Operator packet
    if value_of_bits(bits, 1) == 0:
        sub_packet_length = value_of_bits(bits, 15)
        sub_packet_bits = islice(bits, sub_packet_length)
        sub_packets = list(parse_all_packets(sub_packet_bits))
    else:
        packet_count = value_of_bits(bits, 11)
        sub_packets = [parse_packet(bits) for _ in range(packet_count)]
    return OperatorPacket(ptype, version, sub_packets)


def value_of_bits(bits: Iterable[str], length: int) -> int:
    return int("".join(islice(bits, length)), 2)


def parse_all_packets(bits: Iterable[str]) -> Iterator[Packet]:
    while (packet := parse_packet(bits)) is not None:
        yield packet


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
    assert all(b == "0" for b in bits)  # Have we consumed all the meaningful bits?
    assert packet.sum_of_versions() == expected


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
    ),
)
def test_evaluate(hex, expected):
    bits = generate_bits(hex)
    packet = parse_packet(bits)
    assert all(b == "0" for b in bits)  # Have we consumed all the meaningful bits?
    assert packet.evaluate() == expected


if __name__ == "__main__":
    input_path = get_input_path(16, year=2021)
    with Timer() as t:
        main(input_path, t)
