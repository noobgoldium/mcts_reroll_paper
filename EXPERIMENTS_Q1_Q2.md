# Q1 / Q2 실험 실행 안내

## Q1: 일반도우 → 빛도우 전환 시점 비교

### 정책 정의

- **Q1-A**: 목표 옵션 매칭 슬롯이 1개 이상이면 빛도우로 전환한다.
- **Q1-B**: 목표 옵션 매칭 슬롯이 2개 이상이면 빛도우로 전환한다.
- 매칭 슬롯이 0개이면 빛도우를 사용할 수 없으므로 일반도우를 사용한다.
- Q1-B는 매칭 슬롯이 1개일 때 일반도우를 계속 사용한다. 이때 일반도우가 기존 매칭 슬롯을 잃게 만들 수도 있으며, 이는 비교하려는 정책 차이의 일부다.
- 빛도우는 매칭되지 않은 슬롯 중 하나를 균등 무작위로 선택한다. 매칭 슬롯은 보존한다.
- 목표 달성 또는 합법적 행동이 없어질 때 종료한다. 성공 여부와 자원 소모량은 별도 결과로 기록한다.
- 일반도우가 목표 전환 임계값에 도달하기 전에 고갈되면 빛도우가 합법적인 경우 강제 전환한다. CSV의 `switch_reason=1`이 이 경우다. `switch_reason=0`은 목표 매칭 임계값에 따른 전환이다.

두 정책은 같은 `biscuit_id`의 동일 초기 비스킷을 사용한다. 정책별 난수 스트림은 분리하지만 seed와 biscuit ID로 결정되므로 재현 가능하다. Q1 실행은 CPU worker에 병렬 분배하며 CSV는 biscuit ID 순서로 기록한다.

### 직접 실행

```bash
g++-14 -O3 -march=native -flto -funroll-loops -pipe -std=c++26 -pthread mcts_engine.cpp -o mcts_engine

./mcts_engine \
  --tier 0 --goal 0 \
  --n_starts 100000 --id_start 0 --id_end 100000 \
  --seed 20261123 --q1_compare \
  --q1_out out/q1_policy_tier_0_goal_0.csv

python q1_policy_analysis.py out/q1_policy_tier_0_goal_0.csv \
  --bootstrap 10000 --output out/q1_summary.csv
```

분석 결과에는 두 정책의 성공률, paired 성공률 차이의 95% bootstrap CI, exact McNemar p-value, 일반도우/빛도우 소비량과 리롤 횟수 및 최종 점수의 paired 차이가 포함된다.

### GitHub Actions

`MCTS Matrix Simulation` workflow_dispatch에서 `experiment=q1_compare`를 선택한다. 나머지 tier/goals/chunks/chunk_size/seed 설정은 기존 매트릭스 실행과 같다. 각 matrix artifact에는 해당 청크의 Q1 paired CSV가 들어간다. 모든 CSV를 한 디렉터리에 모은 뒤 `q1_policy_analysis.py`에 전달한다.

## Q2: 슬롯별 빛도우 매칭 점수와 Feynman 기준 비교

### 분석 대상과 이벤트 정의

Q2의 분석 단위는 비스킷 전체의 `ACTION_STOP`이 아니라 **각 슬롯이 목표 조건을 만족해 빛도우 리롤 대상에서 잠기는 순간**이다.

새 엔진 CSV의 슬롯별 필드(`slotN_*`)를 long-form 이벤트 행으로 변환한다.

- `slotN_match_event=1`: 해당 decision row에서 슬롯 N이 unmatched → matched로 전환됨.
- `slotN_match_action=1..4`: 매칭을 발생시킨 빛도우 액션. 액션 ID는 대상 슬롯 번호 + 1이어야 한다.
- `slotN_match_action=0`: 일반도우가 만든 매칭 이벤트. 기록은 보존하지만 Q2 빛도우 곡선에서는 제외한다.
- `slotN_match_x`: 첫 빛도우 리롤 이후 누적 소비량 x. 첫 빛도우로 매칭된 이벤트는 x=1이다.
- `slotN_match_score`: `slot_contribution()`으로 계산한 해당 슬롯의 원점수.
- `slotN_opt_idx`: 옵션별 가중치를 확인해 점수를 정규화하는 데 사용한다.

빛도우 이벤트는 `match_action == slot + 1`, `x >= 0`, `slot_matched == 1`을 모두 만족해야 한다. 이 조건을 어기는 이벤트가 있으면 분석을 중단해 엔진 기록 오류를 숨기지 않는다.

### 점수와 Feynman 기준의 대응

옵션마다 가중치가 다르므로 슬롯 기여 점수의 고정 상한 50으로 나누지 않는다. 엔진의 옵션 가중치 배열을 사용해 다음처럼 정규화한다.

```
slot_score_ratio = slot_match_score / (option_weight[opt_idx] * 10)
n_remaining      = bl_remaining_at_switch - slot_match_x
t_n              = sqrt(n_remaining) / (sqrt(n_remaining) + 1)
```

`slot_score_ratio`는 0~1 범위의 슬롯 품질 척도다. `t_n`은 잔여 빛도우 수에 대응시킨 Feynman 비교 기준이다. 이는 서로 다른 문제를 비교하기 위한 조작적 대응이며, 두 문제가 같은 수학적 문제라는 주장은 아니다.

곡선은 두 관점으로 만든다.

1. **소비량 x 기준:** 빛도우를 x개 소비했을 때 잠긴 슬롯의 평균 정규화 점수와 그 시점의 평균 Feynman 기준.
2. **잔여량 n 기준:** 빛도우가 n개 남았을 때 잠긴 슬롯의 평균 정규화 점수와 Feynman 기준.

각 곡선은 전체 슬롯 통합과 슬롯별 결과를 모두 출력한다. 같은 biscuit episode에서 같은 지점에 여러 이벤트가 있으면 먼저 episode 평균으로 축약한 뒤 episode 단위 bootstrap을 수행해, 이벤트가 많은 비스킷이 CI를 과도하게 지배하지 않게 한다.

### 중요한 해석 제한

- `extract_records()`는 MCTS가 탐색한 모든 trajectory를 CSV에 저장하지 않는다. 대표 best-action 경로를 확률적으로 재생성한 경로에 이벤트를 기록한다. 따라서 결과는 **MCTS 대표 정책 경로에서 관측된 슬롯 매칭 표본**이며, 전체 탐색의 모든 리롤 표본이 아니다.
- 이 분석은 목표 조건을 만족해 잠긴 슬롯의 점수 분포를 기술한다. 모든 가능한 리롤에서의 최적 중단 경계나 인과 효과를 직접 추정하는 것은 아니다.
- Feynman 기준은 비교선이지 정답선이 아니다. 관측 곡선이 기준선과 비슷하더라도 수학적 수렴을 입증했다고 표현하지 않는다.
- `meets_min_episodes=False`인 점은 해당 지점의 독립 episode 수가 부족하다는 뜻이다. 탐색적 결과로만 보고하고 확정적인 곡선 해석에 사용하지 않는다.
- 일반도우로 매칭된 이벤트는 기록되지만 Q2 빛도우 분석에서 제외된다. 일반도우가 매칭을 풀었다가 다시 매칭하는 경우도 이벤트 기록의 의미를 보존한다.

### 실행

새 86열 엔진으로 생성한 `biscuit_tier_*.csv` 파일을 한 디렉터리에 모은다. 이전 50열 CSV와 섞지 않는다.

```bash
python q2_regression.py \
  --csv-dir path/to/mcts_csvs \
  --bootstrap 2000 \
  --min-episodes-per-point 20 \
  --output-prefix out/q2
```

생성 파일:

- `q2_slot_events.csv`: 검증을 통과한 빛도우 슬롯 이벤트 원자료(long format).
- `q2_curve_by_spent.csv`: x별 전체 슬롯 통합 곡선.
- `q2_curve_by_spent_and_slot.csv`: x 및 슬롯별 곡선.
- `q2_curve_by_remaining.csv`: 잔여 빛도우 수 n별 전체 슬롯 통합 곡선.
- `q2_curve_by_remaining_and_slot.csv`: 잔여량 및 슬롯별 곡선.

요약 CSV에는 이벤트 수, 독립 episode 수, 평균 정규화 슬롯 점수, Feynman 기준, 두 값의 차이, episode bootstrap 95% CI 및 최소 episode 수 충족 여부가 포함된다. 표본이 적은 지점도 CSV에 남기되 `meets_min_episodes`로 구분한다.

## 검증 범위와 실제 연구 데이터 구분

CI에서는 Q1의 paired output과 재실행 결정론성, Q1 통계 분석, Q2 회귀/cluster bootstrap을 소규모 표본과 합성 데이터로 검증한다. 합성 Q2 데이터에서 얻은 수치는 분석 코드 검증용일 뿐, 연구 결과로 인용하면 안 된다. 실제 Q1/Q2 결과는 실험용 전체 CSV 생성 후 산출해야 한다.
