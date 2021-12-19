""" Advent of Code 2021, Day 19: https://adventofcode.com/2021/day/19 """
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from io import StringIO
from itertools import takewhile, combinations, product
from pathlib import Path
from typing import Iterator, TextIO, Optional, Sequence

import numpy as np
import pytest

from util import Timer, get_input_path

Vector = tuple[int, int, int]


def main(input_path: Path, timer: Timer):
    with open(input_path) as fp:
        scanner_reports = list(parse_input(fp))
    timer.check("Parse input")

    composite_report = CompositeReport.build_composite_report(scanner_reports)
    timer.check("Orient reports")

    print(len(composite_report.get_all_beacons()))
    timer.check("Part 1")

    max_distance = 0
    for left, right in combinations(composite_report.scanners, 2):
        if (d := calc_manhattan_distance(left, right)) > max_distance:
            max_distance = d
    print(max_distance)
    timer.check("Part 2")


def parse_input(fp: TextIO) -> Iterator[ScannerReport]:
    while (next_report := read_scanner_report(fp)) is not None:
        yield next_report


def read_scanner_report(fp: TextIO) -> Optional[ScannerReport]:
    try:
        header = next(fp)
        scanner_id = int(header.split()[2])
        beacon_lines = takewhile(lambda line: line != "", (line.strip() for line in fp))
        return ScannerReport(
            scanner_id,
            np.array(tuple(tuple(map(int, line.split(","))) for line in beacon_lines)),
        )
    except (StopIteration, IndexError):
        return None


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class Direction(Enum):
    POSITIVE = 0
    NEGATIVE = 1
    ONE_EIGHTY = 2


class ScannerReport:
    ROTATION_MATRICES = {
        (Axis.X, Direction.POSITIVE): np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0))),
        (Axis.X, Direction.NEGATIVE): np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0))),
        (Axis.X, Direction.ONE_EIGHTY): np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1))),
        (Axis.Y, Direction.POSITIVE): np.array(((0, 0, -1), (0, 1, 0), (1, 0, 0))),
        (Axis.Y, Direction.NEGATIVE): np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0))),
        (Axis.Y, Direction.ONE_EIGHTY): np.array(((-1, 0, 0), (0, 1, 0), (0, 0, -1))),
        (Axis.Z, Direction.POSITIVE): np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1))),
        (Axis.Z, Direction.NEGATIVE): np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))),
        (Axis.Z, Direction.ONE_EIGHTY): np.array(((-1, 0, 0), (0, -1, 0), (0, 0, 1))),
    }

    def __init__(self, id: int, beacons: np.ndarray, origin: Vector = (0, 0, 0)):
        self.id = id
        self.beacons = beacons
        self.origin = origin

    def __len__(self):
        return self.beacons.shape[0]

    def rotate(self, axis: Axis, direction: Direction) -> ScannerReport:
        rotation_matrix = self.ROTATION_MATRICES[(axis, direction)]
        return ScannerReport(self.id, self.beacons.dot(rotation_matrix))

    def generate_rotations(self) -> Iterator[ScannerReport]:
        # 4 rotations for each of 6 orientations of the x-axis
        next_variation = self
        for _ in range(2):
            for _ in range(3):
                for _ in range(4):
                    yield next_variation
                    next_variation = next_variation.rotate(Axis.X, Direction.POSITIVE)
                # X->Y, Y->Z, Z->X
                next_variation = next_variation.rotate(
                    Axis.Z, Direction.NEGATIVE
                ).rotate(Axis.Y, Direction.NEGATIVE)
            # X -> -Z, Y -> -Y, Z -> -X
            next_variation = next_variation.rotate(Axis.Z, Direction.ONE_EIGHTY).rotate(
                Axis.Y, Direction.POSITIVE
            )

    def translate(self, offset: tuple[int, int, int]) -> ScannerReport:
        return ScannerReport(
            self.id,
            np.add(self.beacons, offset),
            tuple(a + b for a, b in zip(self.origin, offset)),
        )

    def find_overlap(self, other: ScannerReport) -> Optional[ScannerReport]:
        self_beacon_set = self.get_beacon_set()
        for rotated in other.generate_rotations():
            for self_beacon, other_beacon in product(self.beacons, rotated.beacons):
                offset = tuple(s - o for s, o in zip(self_beacon, other_beacon))
                translated = rotated.translate(offset)
                if len(self_beacon_set & translated.get_beacon_set()) >= 12:
                    return translated
        return None

    def get_beacon_set(self) -> set[Vector]:
        return {tuple(beacon) for beacon in self.beacons}

    def __contains__(self, beacon: Vector) -> bool:
        return beacon in self.get_beacon_set()


class CompositeReport:
    def __init__(self, initial_report: ScannerReport):
        self.scanners: list[ScannerReport] = [initial_report]

    @classmethod
    def build_composite_report(
        cls, scanner_reports: Sequence[ScannerReport]
    ) -> CompositeReport:
        result = cls(scanner_reports[0])

        unmatched_reports = list(scanner_reports[1:])
        attempted_matches = defaultdict(set)
        while len(unmatched_reports) > 0:
            next_report = unmatched_reports.pop(0)
            for report in result.scanners:
                if report.id in attempted_matches[next_report.id]:
                    continue
                attempted_matches[next_report.id].add(report.id)
                if (matched_report := report.find_overlap(next_report)) is not None:
                    result.scanners.append(matched_report)
                    break
            else:
                unmatched_reports.append(next_report)
        return result

    def get_all_beacons(self) -> set[Vector]:
        return {tuple(beacon) for scanner in self.scanners for beacon in scanner.beacons}


def calc_manhattan_distance(left: ScannerReport, right: ScannerReport) -> int:
    return sum(abs(a - b) for a, b in zip(left.origin, right.origin))


SAMPLE_INPUT = """\
--- scanner 0 ---
404,-588,-901
528,-643,409
-838,591,734
390,-675,-793
-537,-823,-458
-485,-357,347
-345,-311,381
-661,-816,-575
-876,649,763
-618,-824,-621
553,345,-567
474,580,667
-447,-329,318
-584,868,-557
544,-627,-890
564,392,-477
455,729,728
-892,524,684
-689,845,-530
423,-701,434
7,-33,-71
630,319,-379
443,580,662
-789,900,-551
459,-707,401

--- scanner 1 ---
686,422,578
605,423,415
515,917,-361
-336,658,858
95,138,22
-476,619,847
-340,-569,-846
567,-361,727
-460,603,-452
669,-402,600
729,430,532
-500,-761,534
-322,571,750
-466,-666,-811
-429,-592,574
-355,545,-477
703,-491,-529
-328,-685,520
413,935,-424
-391,539,-444
586,-435,557
-364,-763,-893
807,-499,-711
755,-354,-619
553,889,-390

--- scanner 2 ---
649,640,665
682,-795,504
-784,533,-524
-644,584,-595
-588,-843,648
-30,6,44
-674,560,763
500,723,-460
609,671,-379
-555,-800,653
-675,-892,-343
697,-426,-610
578,704,681
493,664,-388
-671,-858,530
-667,343,800
571,-461,-707
-138,-166,112
-889,563,-600
646,-828,498
640,759,510
-630,509,768
-681,-892,-333
673,-379,-804
-742,-814,-386
577,-820,562

--- scanner 3 ---
-589,542,597
605,-692,669
-500,565,-823
-660,373,557
-458,-679,-417
-488,449,543
-626,468,-788
338,-750,-386
528,-832,-391
562,-778,733
-938,-730,414
543,643,-506
-524,371,-870
407,773,750
-104,29,83
378,-903,-323
-778,-728,485
426,699,580
-438,-605,-362
-469,-447,-387
509,732,623
647,635,-688
-868,-804,481
614,-800,639
595,780,-596

--- scanner 4 ---
727,592,562
-293,-554,779
441,611,-461
-714,465,-776
-743,427,-804
-660,-479,-426
832,-632,460
927,-485,-438
408,393,-506
466,436,-512
110,16,151
-258,-428,682
-393,719,612
-211,-452,876
808,-476,-593
-575,615,604
-485,667,467
-680,325,-822
-627,-443,-432
872,-547,-609
833,512,582
807,604,487
839,-516,451
891,-625,532
-652,-548,-490
30,-46,-14
"""


@pytest.fixture
def sample_input() -> Iterator[TextIO]:
    with StringIO(SAMPLE_INPUT) as fp:
        yield fp


def test_read_scanner_report(sample_input):
    report = read_scanner_report(sample_input)
    expected_beacons = np.array(
        [
            [404, -588, -901],
            [528, -643, 409],
            [-838, 591, 734],
            [390, -675, -793],
            [-537, -823, -458],
            [-485, -357, 347],
            [-345, -311, 381],
            [-661, -816, -575],
            [-876, 649, 763],
            [-618, -824, -621],
            [553, 345, -567],
            [474, 580, 667],
            [-447, -329, 318],
            [-584, 868, -557],
            [544, -627, -890],
            [564, 392, -477],
            [455, 729, 728],
            [-892, 524, 684],
            [-689, 845, -530],
            [423, -701, 434],
            [7, -33, -71],
            [630, 319, -379],
            [443, 580, 662],
            [-789, 900, -551],
            [459, -707, 401],
        ]
    )
    assert report.id == 0
    assert (report.beacons == expected_beacons).all()


def test_parse_input(sample_input):
    reports = tuple(parse_input(sample_input))
    assert len(reports) == 5


def test_generate_variations(sample_input):
    report = read_scanner_report(sample_input)
    report_variations = tuple(report.generate_rotations())
    assert len(report_variations) == 24
    for left, right in combinations(report_variations, 2):
        assert (left.beacons != right.beacons).any()


def test_generate_variations_1(sample_input):
    common_beacons = (
        (686, 422, 578),
        (605, 423, 415),
        (515, 917, -361),
        (-336, 658, 858),
        (-476, 619, 847),
        (-460, 603, -452),
        (729, 430, 532),
        (-322, 571, 750),
        (-355, 545, -477),
        (413, 935, -424),
        (-391, 539, -444),
        (553, 889, -390),
    )
    # Find the variation of scanner 1 that contains all common beacons
    _, report_1, *_ = parse_input(sample_input)
    for variation in report_1.generate_rotations():
        if all(b in variation for b in common_beacons):
            assert True
            break
    else:
        assert False


def test_find_overlap(sample_input):
    report_0, report_1, *_ = parse_input(sample_input)
    overlapping_report = report_0.find_overlap(report_1)
    assert overlapping_report is not None
    assert overlapping_report.origin == (68, -1246, -43)


def test_composite_report(sample_input):
    scanner_reports = tuple(parse_input(sample_input))
    composite_report = CompositeReport.build_composite_report(scanner_reports)
    assert len(composite_report.get_all_beacons()) == 79


def test_calc_manhattan_distance(sample_input):
    _, _, report_2, report_3, *_ = parse_input(sample_input)
    report_2 = report_2.translate((1105, -1205, 1229))
    report_3 = report_3.translate((-92, -2380, -20))
    assert calc_manhattan_distance(report_2, report_3) == 3621


if __name__ == "__main__":
    input_path = get_input_path(19, year=2021)
    with Timer() as timer:
        main(input_path, timer)
