"""
Paired statistical analysis for the fixed Q1-A/Q1-B policy experiment.

Usage:
  python q1_policy_analysis.py q1_paired.csv --bootstrap 10000
"""
import argparse
import csv
import math
from collections import defaultdict
import numpy as np
import pandas as pd

INITIAL_COLUMNS = [
    f"initial_{name}" for i in range(4)
    for name in (f"opt{i}", f"val{i}")
]


def exact_mcnemar_p(a_only, b_only):
    """Two-sided exact McNemar/binomial test on discordant paired outcomes."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(differences, n_boot, rng):
    values = np.asarray(differences, dtype=float)
    estimate = float(values.mean())
    if n_boot <= 0 or len(values) < 2:
        return estimate, float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="q1_policy_summary.csv")
    args = parser.parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")

    df = pd.read_csv(args.csv_path)
    required = [
        "goal_id", "resource_tier", "biscuit_id", "policy", "goal_achieved",
        "bg_spent", "bl_spent", "reroll_steps", "final_score",
    ] + INITIAL_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"missing Q1 columns: {missing}")

    summaries = []
    rng = np.random.default_rng(args.seed)
    for (tier, goal), part in df.groupby(["resource_tier", "goal_id"]):
        if part.duplicated(["biscuit_id", "policy"]).any():
            raise ValueError(f"duplicate policy row for tier={tier}, goal={goal}")
        wide = part.pivot(index="biscuit_id", columns="policy")
        if not {"Q1-A", "Q1-B"}.issubset(set(wide.columns.get_level_values(1))):
            raise ValueError(f"unpaired Q1 policies for tier={tier}, goal={goal}")
        wide = wide.dropna(subset=[
            ("goal_achieved", "Q1-A"), ("goal_achieved", "Q1-B")
        ])
        if wide.empty:
            raise ValueError(f"no complete pairs for tier={tier}, goal={goal}")

        # Paired initial-state identity check: both policies must start from exactly
        # the same four option/value pairs for every biscuit ID.
        for col in INITIAL_COLUMNS:
            a = wide[(col, "Q1-A")].to_numpy()
            b = wide[(col, "Q1-B")].to_numpy()
            if not np.array_equal(a, b):
                raise ValueError(f"initial biscuit mismatch in {col}")

        a_success = wide[("goal_achieved", "Q1-A")].astype(int).to_numpy()
        b_success = wide[("goal_achieved", "Q1-B")].astype(int).to_numpy()
        a_only = int(np.sum((a_success == 1) & (b_success == 0)))
        b_only = int(np.sum((a_success == 0) & (b_success == 1)))
        delta = b_success - a_success
        rate_delta, rate_lo, rate_hi = paired_bootstrap_ci(delta, args.bootstrap, rng)

        row = {
            "resource_tier": int(tier),
            "goal_id": int(goal),
            "paired_biscuits": len(wide),
            "q1a_success_rate": float(a_success.mean()),
            "q1b_success_rate": float(b_success.mean()),
            "success_rate_delta_b_minus_a": rate_delta,
            "success_delta_ci95_low": rate_lo,
            "success_delta_ci95_high": rate_hi,
            "q1a_only_success": a_only,
            "q1b_only_success": b_only,
            "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
        }

        for metric in ("bg_spent", "bl_spent", "reroll_steps", "final_score"):
            a = wide[(metric, "Q1-A")].astype(float).to_numpy()
            b = wide[(metric, "Q1-B")].astype(float).to_numpy()
            estimate, lo, hi = paired_bootstrap_ci(b - a, args.bootstrap, rng)
            row[f"{metric}_delta_b_minus_a"] = estimate
            row[f"{metric}_delta_ci95_low"] = lo
            row[f"{metric}_delta_ci95_high"] = hi
            row[f"q1a_{metric}_mean"] = float(a.mean())
            row[f"q1b_{metric}_mean"] = float(b.mean())
        summaries.append(row)

    result = pd.DataFrame(summaries)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"saved: {args.output}")
    print("CI: paired biscuit-level bootstrap; success p-value: exact two-sided McNemar test.")


if __name__ == "__main__":
    main()
