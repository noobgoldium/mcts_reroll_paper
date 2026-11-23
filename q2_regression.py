"""
Q2: 잔여 빛도우 수에 따른 경험적 중단 품질 임계값과 Feynman 기준 비교.

추정 대상:
  stop_advantage = Q(STOP) - max_a Q(continue, a)
  stop_advantage가 음수에서 양수로 바뀌는 biscuit_score / 200을 임계값으로 정의.

주의:
- Q값은 Open-loop MCTS 행동 이력 노드의 상태분포 평균이다.
- state_score_mean을 변화시키는 것은 관측 상태에서의 조건부 partial-dependence 분석이다.
  따라서 인과적 중단 경계라고 단정하지 않는다.
- bootstrap CI는 biscuit episode 단위 cluster bootstrap이며, 학습된 모델을 고정한
  조건부 표본 불확실성이다. 모델 재학습 불확실성까지 포함하지 않는다.
- Feynman 기준은 구조적 비교값이지, 비스킷 리롤 문제의 이론적 최적해가 아니다.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
FEATURES = [
    "resource_tier",
    "goal_id",
    "state_bl_mean",
    "state_progress_mean",
    "state_matched_mean",
    "state_score_mean",
]
TARGET = "stop_advantage"


def feynman_threshold(n):
    """PNAS 논문의 균등분포 임계값 sqrt(n) / (sqrt(n) + 1)."""
    n = np.asarray(n, dtype=np.float64)
    return np.sqrt(np.maximum(n, 0.0)) / (np.sqrt(np.maximum(n, 0.0)) + 1.0)


def load_data(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "biscuit_tier_*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV 없음: {csv_dir}/biscuit_tier_*.csv")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    required = FEATURES + [
        "q_stop", "q_continue", "terminal_reason", "state_samples",
        "state_matched_min", "resource_tier", "goal_id", "biscuit_id",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"Q2 필수 컬럼 누락: {missing}. 최신 Open-loop 엔진으로 데이터를 생성해야 합니다."
        )

    df = df.dropna(subset=required).copy()
    df = df[
        (df["terminal_reason"] == 0)
        & (df["state_samples"] > 0)
        & (df["state_matched_min"] >= 2)
        & (df["state_bl_mean"] > 0)
        & (df["q_stop"] >= 0)
        & (df["q_continue"] >= 0)
    ].copy()

    df[TARGET] = df["q_stop"] - df["q_continue"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])
    df["group_id"] = (
        df["resource_tier"].astype(str) + ":"
        + df["goal_id"].astype(str) + ":"
        + df["biscuit_id"].astype(str)
    )
    # 같은 biscuit episode의 동일 상태가 중복 CSV로 들어오지 않도록 한다.
    df = df.drop_duplicates(
        subset=["resource_tier", "goal_id", "biscuit_id", "decision_step"],
        keep="first",
    )
    return df


def estimate_threshold(model, row, feature_columns, grid):
    """예측 stop advantage의 음→양 부호 전환점을 선형 보간한다."""
    candidates = pd.DataFrame(
        [{**row, "state_score_mean": float(score)} for score in grid],
        columns=feature_columns,
    )
    pred = model.predict(candidates)
    crossings = np.flatnonzero((pred[:-1] < 0.0) & (pred[1:] >= 0.0))
    if len(crossings) == 0:
        return np.nan

    i = int(crossings[0])
    x0, x1 = float(grid[i]), float(grid[i + 1])
    y0, y1 = float(pred[i]), float(pred[i + 1])
    if abs(y1 - y0) < 1e-12:
        return x1
    return float(np.clip(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0), x0, x1))


def bootstrap_mean_ci(values, n_boot, rng):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    if len(values) == 1 or n_boot <= 0:
        return estimate, np.nan, np.nan
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-groups-per-bin", type=int, default=20)
    parser.add_argument("--output-prefix", default="q2")
    args = parser.parse_args()

    if args.bootstrap < 0 or args.min_groups_per_bin < 2:
        raise ValueError("--bootstrap은 0 이상, --min-groups-per-bin은 2 이상이어야 합니다.")

    df = load_data(args.csv_dir)
    if df["group_id"].nunique() < 30 or len(df) < 100:
        raise RuntimeError(
            f"Q2 표본 부족: {len(df)}행, {df['group_id'].nunique()}개 episode. "
            "최소 30개 episode와 100개 decision row가 필요합니다."
        )

    X = df[FEATURES].astype(float).reset_index(drop=True)
    y = df[TARGET].astype(float).reset_index(drop=True)
    groups = df["group_id"].reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    pred = model.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx].to_numpy()
    print(f"유효 표본: {len(df):,}행 / {groups.nunique():,} episodes")
    print(f"train: {len(train_idx):,}행 / test: {len(test_idx):,}행")
    print(f"Stop advantage R²: {r2_score(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.5f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.5f}")

    score_grid = np.linspace(0.0, 200.0, 401)
    rows = []
    for i in test_idx:
        row = X.iloc[i].to_dict()
        threshold_score = estimate_threshold(model, row, FEATURES, score_grid)
        if np.isfinite(threshold_score):
            n = float(row["state_bl_mean"])
            rows.append({
                "group_id": groups.iloc[i],
                "n_remaining": n,
                "remaining_bin": int(round(n)),
                "estimated_threshold": threshold_score / 200.0,
                "feynman_threshold": float(feynman_threshold(n)),
            })

    if not rows:
        raise RuntimeError(
            "검증 데이터에서 stop advantage의 음→양 부호 전환이 관측되지 않았습니다. "
            "임계값을 임의로 외삽하지 않고 분석을 중단합니다."
        )

    raw = pd.DataFrame(rows)
    raw["difference"] = raw["estimated_threshold"] - raw["feynman_threshold"]
    raw.to_csv(f"{args.output_prefix}_threshold_estimates.csv", index=False)

    # 여러 decision row가 같은 biscuit episode에서 나오므로 먼저 episode/bin 단위로 집계.
    episode_bin = raw.groupby(
        ["remaining_bin", "group_id"], as_index=False
    ).agg(
        estimated_threshold=("estimated_threshold", "mean"),
        feynman_threshold=("feynman_threshold", "mean"),
        difference=("difference", "mean"),
    )

    rng = np.random.default_rng(SEED)
    summary_rows = []
    for n_bin, part in episode_bin.groupby("remaining_bin"):
        if part["group_id"].nunique() < args.min_groups_per_bin:
            continue
        est, est_lo, est_hi = bootstrap_mean_ci(
            part["estimated_threshold"].to_numpy(), args.bootstrap, rng
        )
        fey, fey_lo, fey_hi = bootstrap_mean_ci(
            part["feynman_threshold"].to_numpy(), args.bootstrap, rng
        )
        diff, diff_lo, diff_hi = bootstrap_mean_ci(
            part["difference"].to_numpy(), args.bootstrap, rng
        )
        summary_rows.append({
            "remaining_light_dough": int(n_bin),
            "episodes": int(part["group_id"].nunique()),
            "estimated_threshold": est,
            "estimated_ci95_low": est_lo,
            "estimated_ci95_high": est_hi,
            "feynman_threshold": fey,
            "feynman_ci95_low": fey_lo,
            "feynman_ci95_high": fey_hi,
            "difference_est_minus_feynman": diff,
            "difference_ci95_low": diff_lo,
            "difference_ci95_high": diff_hi,
        })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError(
            "잔여 빛도우 수별 episode 수가 최소 그룹 수에 미달합니다. "
            "표본을 추가 생성하거나 --min-groups-per-bin을 낮춰야 합니다."
        )
    summary.to_csv(f"{args.output_prefix}_threshold_summary.csv", index=False)

    print(f"임계값 추정 행: {len(raw):,}")
    print(f"독립 episode/bin 표본: {len(episode_bin):,}")
    print("\n잔여 빛도우 수별 임계값 및 episode-cluster bootstrap 95% CI:")
    print(summary.to_string(index=False))
    print(f"\n저장: {args.output_prefix}_threshold_estimates.csv")
    print(f"저장: {args.output_prefix}_threshold_summary.csv")


if __name__ == "__main__":
    main()
