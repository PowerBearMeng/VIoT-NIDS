#!/usr/bin/env python3
"""Convert the original Kitsune two-column label CSV to V4 packet labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="", encoding="utf-8") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.reader(source)
        next(reader)
        writer = csv.writer(target)
        writer.writerow(["frame_number", "binary_label"])
        count = 0
        for row in reader:
            if len(row) < 2:
                continue
            writer.writerow([int(row[0]), int(float(row[1]))])
            count += 1
    print(f"converted packet labels rows={count:,} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
