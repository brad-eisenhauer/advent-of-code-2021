""" Count measurements which are greater than the previous measurement """
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "resources" / "input01.txt"


def main():
    with open(INPUT_FILE) as fp:
        measurements = [int(line) for line in fp.readlines()]

    result = sum(
        1
        for predecessor, successor in zip(measurements, measurements[1:])
        if successor > predecessor
    )
    print(result)


def main2():
    with open(INPUT_FILE) as fp:
        measurements = (int(line) for line in fp.readlines())

        previous_measurement = next(measurements)
        result = 0
        for measurement in measurements:
            if measurement > previous_measurement:
                result += 1
            previous_measurement = measurement

    print(result)


if __name__ == "__main__":
    main()
