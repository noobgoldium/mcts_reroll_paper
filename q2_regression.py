"""
Q2: 잔여 빛도우 수에 따른 MCTS의 경험적 중단 품질 임계값을 추정한다.

주의:
- q_light_avg는 세션 평균이 아니라, 한 open-loop 노드에서 합법적인 빛도우 행동 Q의 평균이다.
- 이 분석은 q_stop - q_continue의 부호가 바뀌는 biscuit_score를 중단 임계값의
  경험적 대리변수로 사용한다.
- 일반도우가 합법적인 상태가 섞이지 않도록 해당 노드의 state_matched_min >= 2를 요구한다.
- Feynman 임계값과의 비교는 구조적 유사성 검토이지, 두 문제의 수학적 동등성 증명이 아니다.
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_DIR = "."
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
    """PNAS 논문의 균등분포 임계값: sqrt(n) / (sqrt(n) + 1)."""
    n = np.asarray(n, dtype=np.float64)
    return np.sqrt(n) / (np.sqrt(n) + 1.0)


def load_data():
    files = glob.glob(os.path.join(CSV_DIR, "biscuit_tier_*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 없음: {CSV_DIR}/biscuit_tier_*.csv")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)

    required = FEATURES + [
        "q_stop", "q_continue", "terminal_reason", "state_samples",
        "state_matched_min", "resource_tier", "goal_id", "biscuit_id",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"현재 CSV 스키마에는 Q2에 필요한 필드가 없습니다: {missing}. "
            "새 MCTS 엔진으로 데이터를 다시 생성해야 합니다."
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    df["group_id"] = (
        df["resource_tier"].astype(str) + ":"
        + df["goal_id"].astype(str) + ":"
        + df["biscuit_id"].astype(str)
    )
    return df


def estimate_threshold(model, row, feature_columns, grid):
    """q_stop - q_continue가 음수에서 0 이상으로 바뀌는 최초 score."""
    candidates = pd.DataFrame(
        [{**row, "state_score_mean": float(score)} for score in grid],
        columns=feature_columns,
    )
    pred = model.predict(candidates)
    crossing = np.flatnonzero(pred >= 0.0)
    if len(crossing) == 0:
        return np.nan
    return float(grid[crossing[0]])


def main():
    df = load_data()
    if len(df) < 100:
        raise RuntimeError(
            f"Q2 분석 표본이 부족합니다: {len(df)}행. "
            "MCTS 탐색량 또는 biscuit 수를 늘려야 합니다."
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
    print(f"표본 수: {len(df):,} | train: {len(train_idx):,} | test: {len(test_idx):,}")
    print(f"Stop advantage R²: {r2_score(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.5f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.5f}")

    # 테스트 표본별로 다른 상태 변수는 고정하고 현재 비스킷 점수만 변화시킨다.
    score_grid = np.linspace(0.0, 200.0, 401)
    rows = []
    for i in test_idx:
        row = X.iloc[i].to_dict()
        threshold = estimate_threshold(model, row, FEATURES, score_grid)
        if np.isfinite(threshold):
            n = float(row["state_bl_mean"])
            rows.append({
                "n_remaining": n,
                "estimated_threshold": threshold / 200.0,
                "feynman_threshold": float(feynman_threshold(n)),
                "absolute_error": abs(
                    threshold / 200.0 - float(feynman_threshold(n))
                ),
            })

    if not rows:
        print("중단 임계값의 부호 교차가 관측되지 않아 Feynman 비교를 수행할 수 없습니다.")
        return

    result = pd.DataFrame(rows)
    print(f"임계값 추정 가능 표본: {len(result):,}")
    print(
        "Feynman 비교 | "
        f"MAE={result['absolute_error'].mean():.5f} | "
        f"RMSE={np.sqrt(np.mean((result['estimated_threshold'] - result['feynman_threshold']) ** 2)):.5f}"
    )
    result["remaining_bin"] = result["n_remaining"].round().astype(int)
    summary = result.groupby("remaining_bin").agg(
        estimated=("estimated_threshold", "mean"),
        feynman=("feynman_threshold", "mean"),
        samples=("estimated_threshold", "size"),
    )
    print("\n잔여 빛도우 수별 임계값:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
