#!/usr/bin/env python3
"""Validate and merge independently generated MCTS chunk CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEY_FIELDS = ("resource_tier", "goal_id", "biscuit_id", "rng_seed", "decision_step")


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list without duplicates")
    return values


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
    total_rows = 0

    for goal in args.goals:
        merged_rows: list[dict[str, str]] = []
        expected_header: list[str] | None = None
        seen_keys: set[tuple[str, ...]] = set()

        for chunk in args.chunks:
            path = args.input_dir / (
                f"biscuit_tier_{args.tier}_goal_{goal}_chunk_{chunk}.csv"
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing or empty chunk CSV: {path}")

            start_id = chunk * args.chunk_size
            end_id = (chunk + 1) * args.chunk_size
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                if not header:
                    raise SystemExit(f"CSV has no header: {path}")
                if expected_header is None:
                    expected_header = header
                    missing = [field for field in KEY_FIELDS if field not in header]
                    if missing:
                        raise SystemExit(f"{path}: missing key columns: {missing}")
                elif header != expected_header:
                    raise SystemExit(f"CSV schema mismatch in {path}")

                chunk_rows = 0
                chunk_biscuit_ids: set[int] = set()
                for row in reader:
                    try:
                        row_tier = int(row["resource_tier"])
                        row_goal = int(row["goal_id"])
                        biscuit_id = int(row["biscuit_id"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise SystemExit(f"Invalid identifying fields in {path}: {exc}") from exc

                    if row_tier != args.tier or row_goal != goal:
                        raise SystemExit(
                            f"Wrong tier/goal in {path}: got tier={row_tier}, goal={row_goal}"
                        )
                    if not start_id <= biscuit_id < end_id:
                        raise SystemExit(
                            f"biscuit_id {biscuit_id} is outside chunk {chunk} range "
                            f"[{start_id}, {end_id}) in {path}"
                        )

                    key = tuple(row[field] for field in KEY_FIELDS)
                    if key in seen_keys:
                        raise SystemExit(f"Duplicate decision key {key} in goal {goal}")
                    seen_keys.add(key)
                    merged_rows.append(row)
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

        output = args.output_dir / f"biscuit_tier_{args.tier}_goal_{goal}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_header)
            writer.writeheader()
            writer.writerows(merged_rows)

        total_rows += len(merged_rows)
        print(
            f"Merged tier={args.tier}, goal={goal}: "
            f"{len(args.chunks)} chunks, {len(merged_rows)} rows -> {output}"
        )

    print(f"Validation and merge complete: {len(args.goals)} files, {total_rows} rows")


if __name__ == "__main__":
    main()
