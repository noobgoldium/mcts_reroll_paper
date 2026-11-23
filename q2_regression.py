"""
Q2: Open-loop MCTS의 STOP/CONTINUE edge-value gap과 Feynman 기준 비교.

중요한 해석 제한:
- q_stop은 STOP edge의 방문 평균, q_continue은 현재 대표 상태에서 합법적인
  계속 행동 중 최고 Q edge의 방문 평균이다.
- Open-loop 노드에서는 STOP edge와 특정 continue edge가 서로 다른 합법성
  조건부 상태 표본을 가질 수 있다. 따라서 gap은 동일 상태의 반사실적 비교가
  아니며, 인과적/이론적 최적 중단 경계가 아니라 기술적(action-value) 지표다.
- ExtraTrees 예측은 계단형이다. 임계값은 부호가 처음 바뀌는 grid 구간의
  상단 grid 점으로 정의하며, 선형 보간으로 가짜 정밀도를 만들지 않는다.
- CI는 고정된 학습 모델을 조건으로 한 episode-cluster bootstrap이다.
  모델 재학습 불확실성은 포함하지 않는다.
- Feynman 식은 구조적 비교 기준이며 두 문제가 동등하다는 주장이 아니다.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
FEATURES = [
    "resource_tier",
    "goal_id",
    "state_bl_mean",
    "state_progress_mean",
    "state_matched_mean",
    "state_score_mean",
]
TARGET = "edge_value_gap"


def feynman_threshold(n):
    """PNAS 논문의 균등분포 임계값 sqrt(n) / (sqrt(n) + 1)."""
    n = np.asarray(n, dtype=np.float64)
    root = np.sqrt(np.maximum(n, 0.0))
    return root / (root + 1.0)


def load_data(csv_dir, min_q_visits=20, min_state_samples=20):
    files = sorted(glob.glob(os.path.join(csv_dir, "biscuit_tier_*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV 없음: {csv_dir}/biscuit_tier_*.csv")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    required = FEATURES + [
        "q_stop", "q_continue", "q_stop_visits", "q_continue_visits",
        "q_continue_action", "terminal_reason", "state_samples",
        "state_matched_min", "resource_tier", "goal_id", "biscuit_id",
        "rng_seed", "decision_step",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"Q2 필수 컬럼 누락: {missing}. 새 엔진으로 데이터를 생성해야 합니다."
        )

    numeric = required
    df = df.dropna(subset=numeric).copy()
    finite_cols = FEATURES + [
        "q_stop", "q_continue", "q_stop_visits", "q_continue_visits",
        "state_samples",
    ]
    finite_mask = np.isfinite(df[finite_cols].to_numpy(dtype=np.float64)).all(axis=1)
    df = df.loc[finite_mask].copy()

    before = len(df)
    df = df[
        (df["terminal_reason"] == 0)
        & (df["state_samples"] >= min_state_samples)
        & (df["q_stop_visits"] >= min_q_visits)
        & (df["q_continue_visits"] >= min_q_visits)
        & (df["state_matched_min"] >= 2)
        & (df["state_bl_mean"] > 0)
        & (df["q_stop"] >= 0)
        & (df["q_continue"] >= 0)
    ].copy()
    print(
        f"방문 수 필터: {before:,}행 → {len(df):,}행 "
        f"(state≥{min_state_samples}, STOP/CONTINUE edge≥{min_q_visits})"
    )

    df[TARGET] = df["q_stop"] - df["q_continue"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    df["group_id"] = (
        df["resource_tier"].astype(str) + ":"
        + df["goal_id"].astype(str) + ":"
        + df["biscuit_id"].astype(str) + ":"
        + df["rng_seed"].astype(str)
    )
    df = df.drop_duplicates(
        subset=["resource_tier", "goal_id", "biscuit_id", "rng_seed", "decision_step"],
        keep="first",
    )
    return df


def estimate_threshold(model, row, feature_columns, grid):
    """
    계단형 모델의 첫 음→양 전환 구간을 반환한다.
    반환값: (상단 grid 점, 구간 하한, 구간 상한); 전환이 없으면 NaN 세 개.
    """
    candidates = pd.DataFrame(
        [{**row, "state_score_mean": float(score)} for score in grid],
        columns=feature_columns,
    )
    pred = np.asarray(model.predict(candidates), dtype=np.float64)
    crossings = np.flatnonzero((pred[:-1] < 0.0) & (pred[1:] >= 0.0))
    if len(crossings) == 0:
        return np.nan, np.nan, np.nan
    i = int(crossings[0])
    lower = float(grid[i])
    upper = float(grid[i + 1])
    # ExtraTrees의 piecewise-constant prediction에 선형 보간을 적용하지 않는다.
    return upper, lower, upper


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
    parser.add_argument("--min-q-visits", type=int, default=20)
    parser.add_argument("--min-state-samples", type=int, default=20)
    parser.add_argument("--output-prefix", default="q2")
    args = parser.parse_args()

    if args.bootstrap < 0:
        raise ValueError("--bootstrap은 0 이상이어야 합니다.")
    if args.min_groups_per_bin < 2:
        raise ValueError("--min-groups-per-bin은 2 이상이어야 합니다.")
    if args.min_q_visits < 1 or args.min_state_samples < 1:
        raise ValueError("방문 수 최소 기준은 1 이상이어야 합니다.")

    df = load_data(
        args.csv_dir,
        min_q_visits=args.min_q_visits,
        min_state_samples=args.min_state_samples,
    )
    if df["group_id"].nunique() < 30 or len(df) < 100:
        raise RuntimeError(
            f"Q2 표본 부족: {len(df)}행, {df['group_id'].nunique()}개 episode. "
            "필터 기준 또는 실험 표본을 점검해야 합니다."
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
    print(f"Edge-value gap R²: {r2_score(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.5f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.5f}")

    score_grid = np.linspace(0.0, 200.0, 401)
    grid_step = float(score_grid[1] - score_grid[0])
    rows = []
    for i in test_idx:
        row = X.iloc[i].to_dict()
        threshold, lower, upper = estimate_threshold(model, row, FEATURES, score_grid)
        if np.isfinite(threshold):
            n = float(row["state_bl_mean"])
            rows.append({
                "group_id": groups.iloc[i],
                "n_remaining": n,
                "remaining_bin": int(round(n)),
                "estimated_threshold": threshold / 200.0,
                "threshold_bracket_low": lower / 200.0,
                "threshold_bracket_high": upper / 200.0,
                "feynman_threshold": float(feynman_threshold(n)),
                "q_stop_visits": int(df.iloc[i]["q_stop_visits"]),
                "q_continue_visits": int(df.iloc[i]["q_continue_visits"]),
                "q_continue_action": int(df.iloc[i]["q_continue_action"]),
            })

    if not rows:
        raise RuntimeError(
            "검증 데이터에서 stop/continue edge-value gap의 음→양 전환이 없습니다. "
            "임계값을 임의로 외삽하지 않고 분석을 중단합니다."
        )

    raw = pd.DataFrame(rows)
    raw["difference"] = raw["estimated_threshold"] - raw["feynman_threshold"]
    raw.to_csv(f"{args.output_prefix}_threshold_estimates.csv", index=False)

    episode_bin = raw.groupby(
        ["remaining_bin", "group_id"], as_index=False
    ).agg(
        estimated_threshold=("estimated_threshold", "mean"),
        threshold_bracket_low=("threshold_bracket_low", "mean"),
        threshold_bracket_high=("threshold_bracket_high", "mean"),
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
            "threshold_bracket_low_mean": float(part["threshold_bracket_low"].mean()),
            "threshold_bracket_high_mean": float(part["threshold_bracket_high"].mean()),
            "threshold_grid_step": grid_step / 200.0,
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
    print(f"임계값 grid 해상도: {grid_step:.3f} score ({grid_step / 200.0:.5f} normalized)")
    print("\n잔여 빛도우 수별 기술적 임계점 및 episode-cluster bootstrap 95% CI:")
    print(summary.to_string(index=False))
    print(f"\n저장: {args.output_prefix}_threshold_estimates.csv")
    print(f"저장: {args.output_prefix}_threshold_summary.csv")


if __name__ == "__main__":
    main()
