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

## Q2: 경험적 중단 임계값과 Feynman 기준 비교

Q2는 각 Open-loop 행동 이력 노드에서 다음 값을 사용한다.

```
stop_advantage = q_stop - q_continue
```

여기서 `q_continue`는 합법적인 계속 행동 중 가장 높은 Q값이다. 예측된 `stop_advantage`가 음수에서 양수로 바뀌는 지점의 `biscuit_score / 200`을 경험적 중단 품질 임계값으로 정의한다. 점수의 이론적 상한은 슬롯별 최대 기여도 50점 × 4슬롯 = 200점이다.

일반도우가 합법적인 상태가 섞이지 않도록 `state_matched_min >= 2`인 노드만 사용한다. 학습/테스트 분리는 biscuit episode 단위 GroupShuffleSplit으로 수행한다. 여러 decision row가 한 비스킷에서 나오는 점을 고려해 최종 CI는 episode 단위 cluster bootstrap으로 계산한다.

Feynman 비교식은 다음과 같다.

```
t_n = sqrt(n) / (sqrt(n) + 1)
```

여기서 n을 잔여 빛도우 수로 대응시키는 것은 비교를 위한 조작적 대응이다. 식당 선택 문제와 비스킷 리롤 문제가 동일한 수학적 문제라는 뜻은 아니다. CI는 고정된 학습 모델을 조건으로 한 episode 표본 불확실성이며, 모델 재학습 불확실성은 포함하지 않는다.

### 실행

최신 MCTS 엔진으로 생성한 `biscuit_tier_*.csv` 파일들을 한 디렉터리에 모은다.

```bash
python q2_regression.py \
  --csv-dir path/to/mcts_csvs \
  --bootstrap 2000 \
  --min-groups-per-bin 20 \
  --output-prefix out/q2
```

다음 두 파일이 생성된다.

- `q2_threshold_estimates.csv`: 테스트 episode의 임계값 추정 행
- `q2_threshold_summary.csv`: 잔여 빛도우 수별 추정 임계값, Feynman 값, 차이 및 95% CI

표본이 부족하거나 검증 데이터에서 음→양 부호 전환이 관측되지 않으면 임의로 외삽하지 않고 오류로 중단한다.

## 검증 범위와 실제 연구 데이터 구분

CI에서는 Q1의 paired output과 재실행 결정론성, Q1 통계 분석, Q2 회귀/cluster bootstrap을 소규모 표본과 합성 데이터로 검증한다. 합성 Q2 데이터에서 얻은 수치는 분석 코드 검증용일 뿐, 연구 결과로 인용하면 안 된다. 실제 Q1/Q2 결과는 실험용 전체 CSV 생성 후 산출해야 한다.
