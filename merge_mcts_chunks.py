#!/usr/bin/env python3
"""Validate and merge MCTS chunk CSV files into one CSV per goal."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list without duplicates")
    return values


def merge_goal(
    *,
    input_dir: Path,
    output_dir: Path,
    tier: int,
    goal: int,
    chunks: list[int],
    chunk_size: int,
    expected_header: list[str] | None,
) -> tuple[int, list[str]]:
    output = output_dir / f"biscuit_tier_{tier}_goal_{goal}_merged.csv"
    temporary_output = output.with_name(output.name + ".tmp")
    goal_rows = 0

    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as output_handle:
            writer: csv.DictWriter | None = None

            for chunk in chunks:
                path = input_dir / f"biscuit_tier_{tier}_goal_{goal}_chunk_{chunk}.csv"
                if not path.is_file() or path.stat().st_size == 0:
                    raise SystemExit(f"Missing or empty chunk CSV: {path}")

                start_id = chunk * chunk_size
                end_id = (chunk + 1) * chunk_size
                with path.open("r", newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    header = reader.fieldnames
                    if not header:
                        raise SystemExit(f"CSV has no header: {path}")

                    if expected_header is None:
                        expected_header = header
                    elif header != expected_header:
                        raise SystemExit(f"CSV schema mismatch in {path}")

                    if writer is None:
                        writer = csv.DictWriter(output_handle, fieldnames=expected_header)
                        writer.writeheader()

                    chunk_rows = 0
                    chunk_biscuit_ids: set[int] = set()
                    for row in reader:
                        try:
                            row_tier = int(row["resource_tier"])
                            row_goal = int(row["goal_id"])
                            biscuit_id = int(row["biscuit_id"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise SystemExit(
                                f"Invalid identifying fields in {path}: {exc}"
                            ) from exc

                        if row_tier != tier or row_goal != goal:
                            raise SystemExit(
                                f"Wrong tier/goal in {path}: "
                                f"got tier={row_tier}, goal={row_goal}"
                            )
                        if not start_id <= biscuit_id < end_id:
                            raise SystemExit(
                                f"biscuit_id {biscuit_id} is outside chunk {chunk} range "
                                f"[{start_id}, {end_id}) in {path}"
                            )

                        # Preserve every observation, including repeated decision keys.
                        writer.writerow(row)
                        chunk_biscuit_ids.add(biscuit_id)
                        chunk_rows += 1

                    expected_ids = set(range(start_id, end_id))
                    missing_ids = expected_ids - chunk_biscuit_ids
                    if missing_ids:
                        preview = sorted(missing_ids)[:10]
                        raise SystemExit(
                            f"Chunk {chunk} is incomplete: {len(missing_ids)} biscuit IDs "
                            f"have no CSV rows; first missing IDs: {preview}"
                        )
                    if chunk_biscuit_ids != expected_ids:
                        raise SystemExit(f"Unexpected biscuit ID coverage in {path}")

                goal_rows += chunk_rows

        if writer is None:
            raise SystemExit(f"No CSV chunks were processed for goal {goal}")
        os.replace(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    return goal_rows, expected_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier", type=int, required=True)
    parser.add_argument("--goals", type=parse_int_list, required=True)
    parser.add_argument("--chunks", type=parse_int_list, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_header: list[str] | None = None
    total_rows = 0

    for goal in args.goals:
        goal_rows, expected_header = merge_goal(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tier=args.tier,
            goal=goal,
            chunks=args.chunks,
            chunk_size=args.chunk_size,
            expected_header=expected_header,
        )
        total_rows += goal_rows
        print(
            f"Validated tier={args.tier}, goal={goal}: "
            f"{len(args.chunks)} chunks, {goal_rows} rows"
        )

    print(
        f"Validation and merge complete: {len(args.goals) * len(args.chunks)} chunks, "
        f"{total_rows} rows -> {args.output_dir} "
        f"({len(args.goals)} goal CSV files)"
    )


if __name__ == "__main__":
    main()
