#!/usr/bin/env python3
"""
Q2: 슬롯별 빛도우 리롤 중단(매칭 잠금) 점수와 Feynman 기준 비교.

분석 단위:
- C++ 기록의 slotN_match_event == 1인 행을 슬롯별 이벤트로 펼친다.
- slotN_match_action이 1~4이고 action == slot index + 1인 이벤트만
  빛도우 리롤로 해당 슬롯이 매칭된 이벤트로 인정한다.
- 일반도우(action 0)로 발생한 매칭은 별도 집계하고 Q2 곡선에서는 제외한다.
- x = slotN_match_x: 첫 빛도우 리롤 이후 누적 빛도우 소비량.
- n_remaining = bl_remaining_at_switch - x.
- 점수 정규화는 slot_contribution의 최대값(weight * 10)을 기준으로 한다.
  따라서 서로 다른 옵션 가중치의 슬롯을 0~1 척도로 비교할 수 있다.

해석 제한:
- 이벤트는 MCTS 전체 탐색의 모든 trajectory가 아니라 extract_records()가
  재생성한 대표 best-action 경로에서 관측된 슬롯 매칭 이벤트다.
- 곡선은 해당 대표 경로에서 목표 조건을 만족해 잠긴 슬롯의 경험적 점수다.
  모든 가능한 리롤의 최적 중단 경계나 인과 효과로 해석하지 않는다.
- Feynman 식은 n_remaining에 대응시킨 구조적 비교 기준이다.
  식당 선택 문제와 비스킷 리롤 문제가 동일하다는 뜻이 아니다.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OPTION_WEIGHTS = np.asarray([1, 1, 1, 5, 1, 5, 5, 1, 1, 1, 5, 5], dtype=float)
IDENTITY = ["resource_tier", "goal_id", "biscuit_id", "rng_seed"]


def feynman_threshold(n):
    """t_n = sqrt(n) / (sqrt(n) + 1), n은 잔여 빛도우 수."""
    n = np.asarray(n, dtype=np.float64)
    root = np.sqrt(np.maximum(n, 0.0))
    return root / (root + 1.0)


def _required_columns():
    required = set(IDENTITY + ["bl_remaining_at_switch", "decision_step"])
    for slot in range(4):
        required.update({
            f"slot{slot}_opt_idx",
            f"slot{slot}_value",
            f"slot{slot}_tier",
            f"slot{slot}_matched",
            f"slot{slot}_match_event",
            f"slot{slot}_match_action",
            f"slot{slot}_match_x",
            f"slot{slot}_match_step",
            f"slot{slot}_match_score",
        })
    return required


def load_slot_events(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "biscuit_tier_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"CSV 없음: {csv_dir}/biscuit_tier_*.csv"
        )

    frames = []
    expected = _required_columns()
    for path in files:
        frame = pd.read_csv(path)
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path}: Q2 슬롯 이벤트 컬럼 누락: {missing}. "
                "새 86열 엔진으로 CSV를 다시 생성해야 합니다."
            )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    event_rows = []
    general_events = 0
    invalid_light_events = []

    for slot in range(4):
        event_col = f"slot{slot}_match_event"
        action_col = f"slot{slot}_match_action"
        x_col = f"slot{slot}_match_x"
        score_col = f"slot{slot}_match_score"
        opt_col = f"slot{slot}_opt_idx"

        marked = df[pd.to_numeric(df[event_col], errors="coerce").fillna(0) == 1]
        for row_index, row in marked.iterrows():
            try:
                action = int(row[action_col])
                matched = int(row[f"slot{slot}_matched"])
                x = int(row[x_col])
                opt_idx = int(row[opt_col])
                raw_score = float(row[score_col])
                switch_remaining = int(row["bl_remaining_at_switch"])
            except (TypeError, ValueError, OverflowError):
                invalid_light_events.append((row_index, slot, "non-numeric event field"))
                continue

            if action == 0:
                general_events += 1
                continue
            if action < 1 or action > 4:
                invalid_light_events.append((row_index, slot, f"unexpected action={action}"))
                continue
            if action != slot + 1:
                invalid_light_events.append(
                    (row_index, slot, f"action={action} does not target slot={slot}")
                )
                continue
            if matched != 1:
                invalid_light_events.append((row_index, slot, "event row is not matched"))
                continue
            if x < 0 or switch_remaining < 0 or x > switch_remaining:
                invalid_light_events.append(
                    (row_index, slot, f"invalid x={x}, switch_remaining={switch_remaining}")
                )
                continue
            if opt_idx < 0 or opt_idx >= len(OPTION_WEIGHTS):
                invalid_light_events.append((row_index, slot, f"invalid opt_idx={opt_idx}"))
                continue
            if not np.isfinite(raw_score):
                invalid_light_events.append((row_index, slot, "non-finite match score"))
                continue

            max_score = OPTION_WEIGHTS[opt_idx] * 10.0
            normalized_score = raw_score / max_score
            if normalized_score < -1e-5 or normalized_score > 1.00001:
                invalid_light_events.append(
                    (row_index, slot, f"normalized score out of range={normalized_score}")
                )
                continue

            event = {name: row[name] for name in IDENTITY}
            event.update({
                "slot": slot,
                "match_action": action,
                "x_light_spent": x,
                "n_light_remaining": switch_remaining - x,
                "match_step": int(row[f"slot{slot}_match_step"]),
                "decision_step": int(row["decision_step"]),
                "opt_idx": opt_idx,
                "slot_value": float(row[f"slot{slot}_value"]),
                "slot_tier": int(row[f"slot{slot}_tier"]),
                "slot_match_score": raw_score,
                "slot_score_max": max_score,
                "slot_score_ratio": float(np.clip(normalized_score, 0.0, 1.0)),
                "source_row": int(row_index),
            })
            event_rows.append(event)

    if invalid_light_events:
        preview = invalid_light_events[:10]
        raise ValueError(
            f"{len(invalid_light_events)}개의 빛도우 매칭 이벤트가 무결성 검사를 "
            f"통과하지 못했습니다. 처음 항목: {preview}"
        )

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise RuntimeError(
            "유효한 빛도우 슬롯 매칭 이벤트가 없습니다. "
            "일반도우 매칭은 제외되며, 새 표본을 생성해야 합니다."
        )

    events["episode_id"] = (
        events["resource_tier"].astype(str) + ":"
        + events["goal_id"].astype(str) + ":"
        + events["biscuit_id"].astype(str) + ":"
        + events["rng_seed"].astype(str)
    )
    print(f"입력 CSV: {len(files)}개 / 원본 행: {len(df):,}")
    print(f"일반도우 매칭 이벤트(곡선 제외): {general_events:,}")
    print(f"유효 빛도우 슬롯 이벤트: {len(events):,}")
    print(f"독립 biscuit episode: {events['episode_id'].nunique():,}")
    return events


def _bootstrap_ci(values, n_boot, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size < 2 or n_boot <= 0:
        return np.nan, np.nan
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def summarize_curve(events, group_columns, min_episodes, n_boot, seed):
    # 동일 episode가 같은 x/slot에 여러 이벤트를 제공해도 먼저 episode 평균으로
    # 축약하여 episode가 많은 쪽에 과도한 가중치가 실리지 않게 한다.
    episode_columns = group_columns + ["episode_id"]
    per_episode = events.groupby(episode_columns, as_index=False).agg(
        slot_score_ratio=("slot_score_ratio", "mean"),
        n_light_remaining=("n_light_remaining", "mean"),
        event_count=("slot_score_ratio", "size"),
    )
    per_episode["feynman_reference"] = feynman_threshold(
        per_episode["n_light_remaining"].to_numpy()
    )
    per_episode["difference"] = (
        per_episode["slot_score_ratio"] - per_episode["feynman_reference"]
    )

    rng = np.random.default_rng(seed)
    output = []
    for key, part in per_episode.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        scores = part["slot_score_ratio"].to_numpy(dtype=float)
        refs = part["feynman_reference"].to_numpy(dtype=float)
        diffs = part["difference"].to_numpy(dtype=float)
        score_lo, score_hi = _bootstrap_ci(scores, n_boot, rng)
        diff_lo, diff_hi = _bootstrap_ci(diffs, n_boot, rng)
        row.update({
            "episodes": int(part["episode_id"].nunique()),
            "events": int(part["event_count"].sum()),
            "mean_slot_score_ratio": float(np.mean(scores)),
            "score_ci95_low": score_lo,
            "score_ci95_high": score_hi,
            "mean_feynman_reference": float(np.mean(refs)),
            "mean_difference_score_minus_feynman": float(np.mean(diffs)),
            "difference_ci95_low": diff_lo,
            "difference_ci95_high": diff_hi,
            "meets_min_episodes": bool(part["episode_id"].nunique() >= min_episodes),
        })
        output.append(row)
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(
        description="86열 MCTS CSV의 슬롯별 빛도우 매칭 점수와 Feynman 기준 비교"
    )
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-episodes-per-point", type=int, default=20)
    parser.add_argument("--output-prefix", default="q2")
    args = parser.parse_args()

    if args.bootstrap < 0:
        parser.error("--bootstrap은 0 이상이어야 합니다.")
    if args.min_episodes_per_point < 2:
        parser.error("--min-episodes-per-point는 2 이상이어야 합니다.")

    events = load_slot_events(args.csv_dir)
    prefix = args.output_prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(f"{prefix}_slot_events.csv", index=False)

    # 전체 슬롯 통합 곡선과 슬롯별 곡선을 함께 생성한다.
    spent_all = summarize_curve(
        events, ["resource_tier", "goal_id", "x_light_spent"],
        args.min_episodes_per_point, args.bootstrap, SEED
    )
    spent_by_slot = summarize_curve(
        events, ["resource_tier", "goal_id", "slot", "x_light_spent"],
        args.min_episodes_per_point, args.bootstrap, SEED
    )
    remaining_all = summarize_curve(
        events, ["resource_tier", "goal_id", "n_light_remaining"],
        args.min_episodes_per_point, args.bootstrap, SEED
    )
    remaining_by_slot = summarize_curve(
        events, ["resource_tier", "goal_id", "slot", "n_light_remaining"],
        args.min_episodes_per_point, args.bootstrap, SEED
    )

    spent_all.to_csv(f"{prefix}_curve_by_spent.csv", index=False)
    spent_by_slot.to_csv(f"{prefix}_curve_by_spent_and_slot.csv", index=False)
    remaining_all.to_csv(f"{prefix}_curve_by_remaining.csv", index=False)
    remaining_by_slot.to_csv(f"{prefix}_curve_by_remaining_and_slot.csv", index=False)

    print("\n빛도우 소비량 x별 전체 슬롯 통합 곡선:")
    print(spent_all.to_string(index=False))
    print("\n잔여 빛도우 n별 전체 슬롯 통합 곡선:")
    print(remaining_all.to_string(index=False))
    print("\n저장 파일:")
    print(f"  {prefix}_slot_events.csv")
    print(f"  {prefix}_curve_by_spent.csv")
    print(f"  {prefix}_curve_by_spent_and_slot.csv")
    print(f"  {prefix}_curve_by_remaining.csv")
    print(f"  {prefix}_curve_by_remaining_and_slot.csv")
    print(
        "\n주의: meets_min_episodes=False인 점은 표본이 부족한 탐색적 결과이며 "
        "확정적인 곡선 해석에 사용하지 마십시오."
    )


if __name__ == "__main__":
    main()
