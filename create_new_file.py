#!/usr/bin/env python3

import argparse
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def main(day: int, year: int):
    env = Environment(loader=FileSystemLoader("./template"))
    t = env.get_template("aoc.py.jinja")

    output_path = Path() / f"aoc{year}" / f"aoc{day:02d}.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        fp.write(t.render(day=day, year=year))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", "-d", type=int, default=date.today().day)
    parser.add_argument("--year", "-y", type=int, default=date.today().year)

    args = parser.parse_args()

    main(**vars(args))
