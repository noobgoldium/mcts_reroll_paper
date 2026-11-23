/**
 * mcts_engine.cpp — 매트릭스 청크 실행용 MCTS 엔진
 *
 * 각 matrix job은 독립된 biscuit_id 범위와 출력 파일을 소유한다.
 * 단일 실행 재개를 위한 CSV 복구/완료 ID 검색은 사용하지 않는다.
 * 작업 분배는 atomic 카운터, 결과 전달은 MPSC 락프리 큐로 처리한다.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <memory>

// ── 캐시 최적화: 벡터 간접 참조를 없애고 CPU 레지스터/L1 캐시에 박히는 정적 배열 사용 ──
static constexpr int N_OPTS = 12;

static constexpr float OPT_LO[N_OPTS] = {
    3.0f, 5.0f, 3.0f, 3.0f, 3.0f, 2.0f, 5.0f, 4.0f, 2.0f, 2.0f, 5.0f, 8.0f
};
static constexpr float OPT_HI[N_OPTS] = {
    7.5f, 7.5f, 15.0f, 10.0f, 7.0f, 6.0f, 10.0f, 10.0f, 5.0f, 5.0f, 15.0f, 15.0f
};
static constexpr float WEIGHTS[N_OPTS] = {
    1.f, 1.f, 1.f, 5.f, 1.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f
};

// 옵션별 값 테이블을 단일 연속 메모리 영역에 배치해 간접 참조와
// 개별 할당을 제거한다. 현재 최대 옵션 범위는 121개 값이다.
static constexpr int MAX_OPT_VALUES = 121;
static std::array<std::array<float, MAX_OPT_VALUES>, N_OPTS> OPT_VALS{};
static std::array<int, N_OPTS> OPT_SIZES{};

void build_opt_vals() {
    for (int i = 0; i < N_OPTS; ++i) {
        float lo = OPT_LO[i], hi = OPT_HI[i];
        int steps = static_cast<int>(std::round((hi - lo) / 0.1f)) + 1;
        OPT_SIZES[i] = steps;
        for (int j = 0; j < steps; ++j)
            OPT_VALS[i][static_cast<size_t>(j)] =
                std::round((lo + static_cast<float>(j) * 0.1f) * 10.f) / 10.f;
    }
}

// ── FastRNG ──────────────────────────────────────────────
struct FastRNG {
    using result_type = uint64_t;
    uint64_t s[2];
    explicit FastRNG(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) {
        s[0] = seed ^ 0x6C0789651D230094ULL;
        s[1] = (seed * 13137ULL) ^ 0x9E3779B97F4A7C15ULL;
        if ((s[0] | s[1]) == 0) s[1] = 1;
    }
    inline result_type operator()() {
        uint64_t s1 = s[0]; const uint64_t s0 = s[1];
        s[0] = s0; s1 ^= s1 << 23;
        s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        return s[1] + s0;
    }
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return UINT64_MAX; }
    inline int next_int(int n) {
        if (n <= 1) return 0;
        uint32_t r = static_cast<uint32_t>(operator()());
        return static_cast<int>((static_cast<uint64_t>(r) * static_cast<uint64_t>(n)) >> 32);
    }
};
thread_local FastRNG tl_rng{ std::random_device{}() };

inline uint64_t mix_seed(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ── 슬롯 / 비스킷 / 목표 ──────────────────────────────────
struct Slot { int opt_idx; float val; };
using Biscuit = std::array<Slot, 4>;

struct GoalEntry { int opt_idx, required, min_tier; };
using Goal = std::vector<GoalEntry>;

// 핫스팟 함수: 벡터 우회 없이 곧바로 정적 배열을 참조하여 캐시 미스 제로화
inline int get_tier(int opt_idx, float val) {
    float lo = OPT_LO[opt_idx];
    float hi = OPT_HI[opt_idx];
    if (hi <= lo) return 0;
    if (val >= hi - 1e-5f) return 5;
    float ratio = (val - lo) / (hi - lo);
    if (ratio >= 0.99f) return 5;
    if (ratio >= 0.90f) return 4;
    if (ratio >= 0.75f) return 3;
    if (ratio >= 0.50f) return 2;
    return 1;
}

inline std::array<bool, 4> get_matched_slots(const Biscuit& b, const Goal& goal) {
    std::array<bool, 4> matched = { false, false, false, false };
    for (const auto& g : goal) {
        int needed = g.required;
        for (int i = 0; i < 4; ++i) {
            if (matched[i]) continue;
            if (b[i].opt_idx == g.opt_idx &&
                get_tier(b[i].opt_idx, b[i].val) >= g.min_tier) {
                matched[i] = true;
                if (--needed <= 0) break;
            }
        }
    }
    return matched;
}

inline bool check_goal(const Biscuit& b, const Goal& goal) {
    auto matched = get_matched_slots(b, goal);
    int total_required = 0;
    for (const auto& g : goal) total_required += g.required;
    int count = 0;
    for (bool m : matched) if (m) count++;
    return count >= total_required;
}

inline float goal_progress(const Biscuit& b, const Goal& goal) {
    int total_required = 0;
    for (const auto& g : goal) total_required += g.required;
    if (total_required == 0) return 1.0f;
    auto matched = get_matched_slots(b, goal);
    int count = 0;
    for (bool m : matched) if (m) count++;
    return static_cast<float>(count) / static_cast<float>(total_required);
}

inline float slot_contribution(int opt_idx, float val) {
    float lo = OPT_LO[opt_idx];
    float hi = OPT_HI[opt_idx];
    float norm = (hi > lo) ? (val - lo) / (hi - lo) : 1.0f;
    return WEIGHTS[opt_idx] * norm * 10.0f;
}

inline float biscuit_score(const Biscuit& b) {
    float t = 0.0f;
    for (const auto& s : b) t += slot_contribution(s.opt_idx, s.val);
    return t;
}

inline Slot sample_slot(bool has_attribute_damage) {
    const int option_count = has_attribute_damage ? N_OPTS : N_OPTS - 1;
    const int opt = tl_rng.next_int(option_count);
    const int idx = tl_rng.next_int(OPT_SIZES[opt]);
    return { opt, OPT_VALS[opt][static_cast<size_t>(idx)] };
}

inline Biscuit reroll_general(const Biscuit&, bool has_attribute_damage) {
    Biscuit nb;
    for (auto& s : nb) s = sample_slot(has_attribute_damage);
    return nb;
}

inline Biscuit reroll_light(const Biscuit& b, int pos, bool has_attribute_damage) {
    Biscuit nb = b;
    nb[static_cast<size_t>(pos)] = sample_slot(has_attribute_damage);
    return nb;
}

inline float compute_reward(const Biscuit& b, const Goal& goal) {
    const float prog = goal_progress(b, goal);
    const float score_norm = biscuit_score(b) / 200.0f;
    const float prog_weight = std::pow(prog * 2.0f, 3.0f);
    return prog_weight + score_norm * 0.1f;
}

// ── MCTS ──────────────────────────────────────────────────
static constexpr int ACTION_GENERAL = 0;
static constexpr int ACTION_LIGHT_FIRST = 1;
static constexpr int ACTION_LIGHT_LAST = 4;
static constexpr int ACTION_STOP = 5;
static constexpr int N_ACTIONS = 6;

// 행동 제약은 트리 탐색과 롤아웃에서 공유한다.
// 매칭 0~1개: 일반도우 허용, 매칭 2개 이상: 일반도우 금지.
// 빛도우는 매칭되지 않은 슬롯에만 사용한다.
inline int count_matched_slots(const std::array<bool, 4>& matched) {
    int count = 0;
    for (bool m : matched) if (m) ++count;
    return count;
}

inline bool is_action_legal(
    const Biscuit& biscuit, int bg, int bl, const Goal& goal, int action)
{
    if (action < 0 || action >= N_ACTIONS) return false;

    const auto matched = get_matched_slots(biscuit, goal);
    const int matched_count = count_matched_slots(matched);

    if (action == ACTION_STOP) return true;
    if (action == ACTION_GENERAL)
        return bg >= 1 && matched_count < 2;
    const int pos = action - ACTION_LIGHT_FIRST;
    return bl >= 1 && pos >= 0 && pos < 4 &&
        matched_count > 0 && !matched[static_cast<size_t>(pos)];
}

// Open-loop MCTS: 노드는 고정된 비스킷 상태가 아니라 행동 이력을 나타낸다.
// 확률적 행동을 선택할 때마다 현재 표본 상태에 새 리롤 결과를 적용한다.
// 따라서 한 행동 간선의 Q는 해당 행동 이력 아래 여러 전이 결과의 평균이다.
struct Node {
    std::array<int, N_ACTIONS> children{ -1,-1,-1,-1,-1,-1 };
    float value = 0.0f;
    int visits = 0;
    int parent = -1;
    int action = -1;

    // State distribution observed at this open-loop action-history node.
    uint64_t state_samples = 0;
    int state_matched_min = 5, state_matched_max = 0;
    double progress_sum = 0.0, matched_sum = 0.0;
    double bg_sum = 0.0, bl_sum = 0.0, score_sum = 0.0;
};

struct Record {
    std::array<int, 5> tier_counts{};
    std::array<int, 12> opt_max_tier{};
    float progress = 0.0f, best_q = 0.0f, q_general = -1.0f, q_light_avg = -1.0f;
    int bg = 0, bl = 0, best_action = -1;
    int goal_id = -1;
    int resource_tier = -1;
    int biscuit_id = -1;
    int matched_at_switch = -1;
    int bl_remaining_at_switch = -1;
    float q_stop = -1.0f, q_continue = -1.0f, score = 0.0f;
    int matched_count = 0, goal_achieved = 0;
    int bg_spent = 0, bl_spent = 0, decision_step = 0;
    int terminal_reason = 0; // 0=decision, 1=voluntary stop, 2=no legal continuation
    uint64_t rng_seed = 0;
    uint64_t state_samples = 0;
    int state_matched_min = -1, state_matched_max = -1;
    double state_progress_mean = 0.0, state_matched_mean = 0.0;
    double state_bg_mean = 0.0, state_bl_mean = 0.0, state_score_mean = 0.0;
};

struct MCTSTree {
    std::vector<Node>& nodes;
    const Goal& goal;
    int max_rollout;
    int initial_bg, initial_bl;
    bool has_attribute_damage;
    Biscuit initial_biscuit;

    MCTSTree(std::vector<Node>& pool, const Biscuit& init,
        int bg, int bl, const Goal& g, int mr)
        : nodes(pool), goal(g), max_rollout(mr), initial_bg(bg), initial_bl(bl),
          has_attribute_damage(std::any_of(g.begin(), g.end(),
              [](const GoalEntry& entry) { return entry.opt_idx == 11; })),
          initial_biscuit(init)
    {
        nodes.emplace_back(); // root = empty action history
    }

    int new_node(int parent, int action) {
        Node n;
        n.parent = parent;
        n.action = action;
        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
    }

    inline float ucb1(int parent, int action, float log_parent_visits,
        float c = 0.70710678f) const
    {
        const int ci = nodes[parent].children[action];
        if (ci < 0) return std::numeric_limits<float>::infinity();
        const Node& child = nodes[ci];
        if (child.visits == 0) return std::numeric_limits<float>::infinity();
        return child.value / static_cast<float>(child.visits) +
            c * std::sqrt(log_parent_visits / static_cast<float>(child.visits));
    }

    void observe_state(int idx, const Biscuit& b, int bg, int bl) {
        Node& n = nodes[idx];
        ++n.state_samples;
        const int matched = count_matched_slots(get_matched_slots(b, goal));
        n.state_matched_min = std::min(n.state_matched_min, matched);
        n.state_matched_max = std::max(n.state_matched_max, matched);
        n.progress_sum += goal_progress(b, goal);
        n.matched_sum += matched;
        n.bg_sum += bg;
        n.bl_sum += bl;
        n.score_sum += biscuit_score(b);
    }

    void fill_state_stats(Record& rec, const Node& n) const {
        rec.state_samples = n.state_samples;
        rec.state_matched_min = n.state_matched_min;
        rec.state_matched_max = n.state_matched_max;
        if (n.state_samples == 0) return;
        const double inv = 1.0 / static_cast<double>(n.state_samples);
        rec.state_progress_mean = n.progress_sum * inv;
        rec.state_matched_mean = n.matched_sum * inv;
        rec.state_bg_mean = n.bg_sum * inv;
        rec.state_bl_mean = n.bl_sum * inv;
        rec.state_score_mean = n.score_sum * inv;
    }

    inline void apply_action(Biscuit& b, int& bg, int& bl, int action) const {
        if (action == ACTION_GENERAL) {
            b = reroll_general(b, has_attribute_damage);
            --bg;
        } else if (action >= ACTION_LIGHT_FIRST && action <= ACTION_LIGHT_LAST) {
            b = reroll_light(b, action - ACTION_LIGHT_FIRST, has_attribute_damage);
            --bl;
        }
    }

    float rollout(Biscuit b, int bg, int bl, bool stopped = false) const {
        if (stopped) return compute_reward(b, goal);

        for (int step = 0; step < max_rollout; ++step) {
            std::array<int, 5> legal{};
            int legal_count = 0;
            if (is_action_legal(b, bg, bl, goal, ACTION_GENERAL))
                legal[legal_count++] = ACTION_GENERAL;
            for (int a = ACTION_LIGHT_FIRST; a <= ACTION_LIGHT_LAST; ++a)
                if (is_action_legal(b, bg, bl, goal, a))
                    legal[legal_count++] = a;
            if (legal_count == 0) break;

            // Rollout policy is deliberately documented as a random-continuation baseline.
            const int action = legal[tl_rng.next_int(legal_count)];
            apply_action(b, bg, bl, action);
        }
        return compute_reward(b, goal);
    }

    void backprop(const std::vector<int>& path, float reward) {
        for (int idx : path) {
            ++nodes[idx].visits;
            nodes[idx].value += reward;
        }
    }

    void run(int n_sim) {
        for (int sim = 0; sim < n_sim; ++sim) {
            // Every simulation starts from the same root state, but samples fresh
            // stochastic outcomes along the selected open-loop action sequence.
            Biscuit b = initial_biscuit;
            int bg = initial_bg, bl = initial_bl;
            std::vector<int> path;
            path.reserve(64);
            int idx = 0;
            path.push_back(idx);
            bool stopped = false;
            bool expanded = false;

            while (!stopped) {
                observe_state(idx, b, bg, bl);
                std::array<int, N_ACTIONS> legal{};
                int legal_count = 0;
                for (int a = 0; a < N_ACTIONS; ++a)
                    if (is_action_legal(b, bg, bl, goal, a))
                        legal[legal_count++] = a;
                if (legal_count == 0) break;

                const float log_parent = std::log(
                    static_cast<float>(std::max(1, nodes[idx].visits)));
                float best_u = -std::numeric_limits<float>::infinity();
                std::array<int, N_ACTIONS> ties{};
                int tie_count = 0;

                for (int k = 0; k < legal_count; ++k) {
                    const int a = legal[k];
                    const int ci = nodes[idx].children[a];
                    const float u = (ci < 0)
                        ? std::numeric_limits<float>::infinity()
                        : ucb1(idx, a, log_parent);
                    if (u > best_u + 1e-7f) {
                        best_u = u;
                        ties[0] = a;
                        tie_count = 1;
                    } else if (std::abs(u - best_u) <= 1e-7f) {
                        ties[tie_count++] = a;
                    }
                }

                const int action = ties[tl_rng.next_int(tie_count)];
                int child = nodes[idx].children[action];
                if (child < 0) {
                    child = new_node(idx, action);
                    nodes[idx].children[action] = child;
                    expanded = true;
                }

                path.push_back(child);
                if (action == ACTION_STOP) {
                    observe_state(child, b, bg, bl);
                    stopped = true;
                    idx = child;
                    break;
                }

                // Crucial: do not store/replay one child biscuit. Resample this
                // action's outcome on every traversal of this action-history node.
                apply_action(b, bg, bl, action);
                idx = child;

                if (expanded) {
                    observe_state(idx, b, bg, bl);
                    break;
                }
            }

            const float reward = rollout(b, bg, bl, stopped);
            backprop(path, reward);
        }
    }

    void fill_state_record(Record& rec, const Biscuit& b, int bg, int bl,
        int depth, int terminal_reason) const
    {
        rec.progress = goal_progress(b, goal);
        rec.bg = bg;
        rec.bl = bl;
        rec.score = biscuit_score(b);
        rec.matched_count = count_matched_slots(get_matched_slots(b, goal));
        rec.goal_achieved = check_goal(b, goal) ? 1 : 0;
        rec.bg_spent = initial_bg - bg;
        rec.bl_spent = initial_bl - bl;
        rec.decision_step = depth;
        rec.terminal_reason = terminal_reason;
        for (const auto& slot : b) {
            const int tier = get_tier(slot.opt_idx, slot.val);
            if (tier >= 1 && tier <= 5)
                rec.tier_counts[static_cast<size_t>(tier - 1)]++;
            int& mx = rec.opt_max_tier[static_cast<size_t>(slot.opt_idx)];
            if (tier > mx) mx = tier;
        }
    }

    std::vector<Record> extract_records(int max_depth = 200) {
        std::vector<Record> recs;
        Biscuit b = initial_biscuit;
        int bg = initial_bg, bl = initial_bl;
        int idx = 0;
        int switch_matched = -1, switch_bl_remaining = -1;

        for (int depth = 0; depth < max_depth; ++depth) {
            Record rec;
            fill_state_record(rec, b, bg, bl, depth, 0);

            const Node& node = nodes[idx];
            fill_state_stats(rec, node);
            int best_action = -1;
            float best_q = -std::numeric_limits<float>::infinity();
            float q_light_sum = 0.0f;
            int q_light_count = 0;

            for (int a = 0; a < N_ACTIONS; ++a) {
                if (!is_action_legal(b, bg, bl, goal, a)) continue;
                const int ci = node.children[a];
                if (ci < 0 || nodes[ci].visits == 0) continue;
                const float q = nodes[ci].value / static_cast<float>(nodes[ci].visits);
                if (q > best_q) { best_q = q; best_action = a; }
                if (a == ACTION_STOP) rec.q_stop = q;
                else {
                    if (q > rec.q_continue) rec.q_continue = q;
                    if (a == ACTION_GENERAL) rec.q_general = q;
                    else { q_light_sum += q; ++q_light_count; }
                }
            }
            if (q_light_count > 0)
                rec.q_light_avg = q_light_sum / static_cast<float>(q_light_count);

            if (best_action < 0) break;
            rec.best_action = best_action;
            rec.best_q = best_q;
            recs.push_back(rec);

            if (best_action == ACTION_STOP) {
                Record terminal;
                fill_state_record(terminal, b, bg, bl, depth + 1, 1);
                fill_state_stats(terminal, nodes[idx]);
                recs.push_back(terminal);
                break;
            }

            if (best_action >= ACTION_LIGHT_FIRST &&
                best_action <= ACTION_LIGHT_LAST && switch_matched < 0) {
                switch_matched = rec.matched_count;
                switch_bl_remaining = bl;
            }

            apply_action(b, bg, bl, best_action);
            idx = node.children[best_action];
            if (idx < 0) break;

            const auto matched = get_matched_slots(b, goal);
            bool has_legal_continue =
                is_action_legal(b, bg, bl, goal, ACTION_GENERAL);
            for (int a = ACTION_LIGHT_FIRST; a <= ACTION_LIGHT_LAST; ++a)
                has_legal_continue = has_legal_continue ||
                    is_action_legal(b, bg, bl, goal, a);
            if (!has_legal_continue) {
                Record terminal;
                fill_state_record(terminal, b, bg, bl, depth + 1, 2);
                fill_state_stats(terminal, nodes[idx]);
                recs.push_back(terminal);
                break;
            }
        }

        if (switch_matched >= 0) {
            for (auto& rec : recs) {
                rec.matched_at_switch = switch_matched;
                rec.bl_remaining_at_switch = switch_bl_remaining;
            }
        }
        return recs;
    }
};

// ── 고성능 락프리 링버퍼 (MPSC) ────────────────────────────
template <typename T, size_t Capacity>
class LockFreeMPSCQueue {
private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    Cell buffer[Capacity];

    // pop()은 I/O 스레드 하나만 호출한다. 소비자 전용 head는 비원자 변수로 둔다.
    // 여러 Worker가 공유하는 tail은 원자성을 유지한다.
    size_t head = 0;
    alignas(64) std::atomic<size_t> tail{0};

public:
    LockFreeMPSCQueue() {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool push(const T& data) {
        Cell* cell;
        size_t t = tail.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer[t & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(t);
            if (diff == 0) {
                if (tail.compare_exchange_weak(t, t + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                t = tail.load(std::memory_order_relaxed);
            }
        }
        cell->data = data;
        cell->sequence.store(t + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& data) {
        const size_t h = head;
        Cell* cell = &buffer[h & (Capacity - 1)];
        const size_t seq = cell->sequence.load(std::memory_order_acquire);
        // 단일 소비자이므로 head 경합/재시도는 존재하지 않는다.
        if (seq != h + 1) return false;

        data = cell->data;
        cell->sequence.store(h + Capacity, std::memory_order_release);
        head = h + 1;
        return true;
    }
};

// ══════════════════════════════════════════════════════════
//  이하 실행/분산/IO 계층 (MCTS 시뮬 로직과 무관)
// ══════════════════════════════════════════════════════════

// ── CSV 헤더 ──────────────────────────────────────────────
static const std::string CSV_HEADER =
"tier_0,tier_1,tier_2,tier_3,tier_4,progress,budget_g,budget_l,"
"opt0_tier,opt1_tier,opt2_tier,opt3_tier,opt4_tier,opt5_tier,"
"opt6_tier,opt7_tier,opt8_tier,opt9_tier,opt10_tier,opt11_tier,"
"best_action,best_q,q_general,q_light_avg,"
"goal_id,resource_tier,biscuit_id,matched_at_switch,bl_remaining_at_switch,"
"q_stop,q_continue,biscuit_score,matched_count,goal_achieved,"
"bg_spent,bl_spent,decision_step,terminal_reason,rng_seed,"
"state_samples,state_progress_mean,state_matched_mean,state_bg_mean,state_bl_mean,state_score_mean,"
"state_matched_min,state_matched_max\n";

static constexpr int EXIT_INTERRUPTED = 75;       // SIGINT/SIGTERM 안전 종료

// 종료 요청 플래그 (SIGINT/SIGTERM 핸들러에서 설정)
static std::atomic<bool> g_stop{ false };
static_assert(std::atomic<bool>::is_always_lock_free, "signal handler requires lock-free atomic");
extern "C" void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

std::vector<Biscuit> generate_inits(const Biscuit& base, int id_start, int id_end,
    bool has_attribute_damage, uint64_t base_seed) {
    std::vector<Biscuit> inits(static_cast<size_t>(id_end - id_start));
    for (int local = 0; local < id_end - id_start; ++local) {
        const int id = id_start + local;
        tl_rng.reseed(mix_seed(base_seed ^ static_cast<uint64_t>(id) ^ 0x494E495449414CULL));
        if (id % 3 == 0) {
            for (int p = 0; p < 4; ++p) {
                int opt = tl_rng.next_int(has_attribute_damage ? N_OPTS : N_OPTS - 1);
                int vidx = tl_rng.next_int(OPT_SIZES[opt]);
                inits[static_cast<size_t>(local)][p] = { opt, OPT_VALS[opt][vidx] };
            }
        }
        else {
            inits[static_cast<size_t>(local)] = base;
            int nr = tl_rng.next_int(3);
            for (int j = 0; j < nr; ++j) {
                int pos = tl_rng.next_int(4);
                int opt = tl_rng.next_int(has_attribute_damage ? N_OPTS : N_OPTS - 1);
                int vidx = tl_rng.next_int(OPT_SIZES[opt]);
                inits[static_cast<size_t>(local)][pos] = { opt, OPT_VALS[opt][vidx] };
            }
        }
    }
    return inits;
}


struct Q1PolicyResult {
    int policy = 0;
    int switch_matched = -1;
    int switch_bl_remaining = -1;
    int switch_reason = 0; // 0=threshold, 1=forced by general-dough exhaustion
    int achieved = 0;
    int terminal_reason = 0; // 1=goal, 2=no legal action, 3=no unmatched slot
    int bg_spent = 0, bl_spent = 0, steps = 0;
    Biscuit final_biscuit{};
    int final_bg = 0, final_bl = 0;
};

Q1PolicyResult simulate_q1_policy(
    const Biscuit& initial, const Goal& goal, int initial_bg, int initial_bl,
    bool has_attribute_damage, int switch_threshold, uint64_t policy_seed)
{
    Q1PolicyResult result;
    result.policy = switch_threshold;
    result.final_biscuit = initial;
    result.final_bg = initial_bg;
    result.final_bl = initial_bl;
    tl_rng.reseed(policy_seed);

    bool switched = false;
    const int max_steps = initial_bg + initial_bl + 1;
    for (int step = 0; step < max_steps; ++step) {
        Biscuit& b = result.final_biscuit;
        int& bg = result.final_bg;
        int& bl = result.final_bl;

        if (check_goal(b, goal)) {
            result.achieved = 1;
            result.terminal_reason = 1;
            break;
        }

        const int matched_count =
            count_matched_slots(get_matched_slots(b, goal));

        if (!switched) {
            if (matched_count >= switch_threshold) {
                switched = true;
                result.switch_matched = matched_count;
                result.switch_bl_remaining = bl;
                result.switch_reason = 0;
            } else if (bg > 0 && matched_count < 2) {
                b = reroll_general(b, has_attribute_damage);
                --bg;
                ++result.bg_spent;
                ++result.steps;
                continue;
            } else if (matched_count > 0 && bl > 0) {
                // General dough is exhausted before the target threshold.
                // Switch to light dough rather than silently changing policy.
                switched = true;
                result.switch_matched = matched_count;
                result.switch_bl_remaining = bl;
                result.switch_reason = 1;
            } else {
                result.terminal_reason = 2;
                break;
            }
        }

        if (bl <= 0 || matched_count <= 0) {
            result.terminal_reason = 2;
            break;
        }

        const auto matched = get_matched_slots(b, goal);
        std::array<int, 4> candidates{};
        int candidate_count = 0;
        for (int pos = 0; pos < 4; ++pos)
            if (!matched[static_cast<size_t>(pos)])
                candidates[candidate_count++] = pos;

        if (candidate_count == 0) {
            result.achieved = check_goal(b, goal) ? 1 : 0;
            result.terminal_reason = result.achieved ? 1 : 3;
            break;
        }

        const int pos = candidates[tl_rng.next_int(candidate_count)];
        b = reroll_light(b, pos, has_attribute_damage);
        --bl;
        ++result.bl_spent;
        ++result.steps;
    }

    if (check_goal(result.final_biscuit, goal)) {
        result.achieved = 1;
        result.terminal_reason = 1;
    } else if (result.terminal_reason == 0) {
        result.terminal_reason = 2;
    }
    return result;
}

int run_q1_compare(
    const Goal& goal, int goal_id, int resource_tier, int initial_bg, int initial_bl,
    int id_start, int id_end, uint64_t base_seed, const std::string& output_path)
{
    const bool has_attribute_damage = std::any_of(goal.begin(), goal.end(),
        [](const GoalEntry& entry) { return entry.opt_idx == 11; });
    const Biscuit base = {
        Slot{ 0, 3.0f }, Slot{ 1, 5.0f }, Slot{ 3, 3.0f }, Slot{ 4, 3.0f }
    };
    const auto initials = generate_inits(
        base, id_start, id_end, has_attribute_damage, base_seed);

    std::ofstream out(output_path, std::ios::trunc);
    if (!out) {
        std::cerr << "[Error] Q1 출력 파일 생성 실패: " << output_path << "\\n";
        return 1;
    }
    out << "goal_id,resource_tier,biscuit_id,base_seed,policy,initial_matched,"
           "switch_threshold,switch_matched,switch_bl_remaining,switch_reason,"
           "goal_achieved,terminal_reason,bg_initial,bl_initial,bg_spent,bl_spent,"
           "bg_remaining,bl_remaining,reroll_steps,initial_score,final_score,"
           "initial_opt0,initial_val0,initial_opt1,initial_val1,initial_opt2,initial_val2,"
           "initial_opt3,initial_val3,final_opt0,final_val0,final_opt1,final_val1,"
           "final_opt2,final_val2,final_opt3,final_val3\\n";

    for (int local = 0; local < id_end - id_start; ++local) {
        if (g_stop.load(std::memory_order_relaxed)) return EXIT_INTERRUPTED;
        const int id = id_start + local;
        const Biscuit& initial = initials[static_cast<size_t>(local)];
        const int initial_matched =
            count_matched_slots(get_matched_slots(initial, goal));
        const uint64_t init_seed = mix_seed(
            base_seed ^ static_cast<uint64_t>(id) ^ 0x494E495449414CULL);

        for (int threshold : { 1, 2 }) {
            const uint64_t policy_seed = mix_seed(
                base_seed ^ static_cast<uint64_t>(id) ^
                (threshold == 1 ? 0x513141ULL : 0x513142ULL));
            const auto r = simulate_q1_policy(
                initial, goal, initial_bg, initial_bl,
                has_attribute_damage, threshold, policy_seed);

            out << goal_id << ',' << resource_tier << ',' << id << ','
                << init_seed << ',' << (threshold == 1 ? "Q1-A" : "Q1-B") << ','
                << initial_matched << ',' << threshold << ','
                << r.switch_matched << ',' << r.switch_bl_remaining << ','
                << r.switch_reason << ',' << r.achieved << ','
                << r.terminal_reason << ',' << initial_bg << ',' << initial_bl << ','
                << r.bg_spent << ',' << r.bl_spent << ','
                << r.final_bg << ',' << r.final_bl << ',' << r.steps << ','
                << biscuit_score(initial) << ',' << biscuit_score(r.final_biscuit);

            for (const auto& slot : initial)
                out << ',' << slot.opt_idx << ',' << slot.val;
            for (const auto& slot : r.final_biscuit)
                out << ',' << slot.opt_idx << ',' << slot.val;
            out << '\\n';
        }
    }
    out.flush();
    if (!out) {
        std::cerr << "[Error] Q1 CSV 기록 실패: " << output_path << "\\n";
        return 1;
    }
    std::cout << "[Q1] paired policies written: " << output_path << "\\n";
    return 0;
}

struct MCTSConfig {
    Biscuit init_biscuit = { Slot{ 0, 3.0f }, Slot{ 1, 5.0f }, Slot{ 3, 3.0f }, Slot{ 4, 3.0f } };
    Goal goal = { { 3, 2, 3 },{ 4, 2, 3 } };
    int budget_g = 90000;
    int budget_l = 3000;
    int n_starts = 100000;
    int n_sim = 500;
    int max_rollout = 300;
    uint64_t seed = 20261123ULL;
    int goal_id = 0;
    int resource_tier = 0;
    int id_start = 0;      // 처리할 biscuit_id 범위 [id_start, id_end)
    int id_end = 100000;
    std::string output_path = "";
};

// 반환: 0 = 범위 전체 완료, EXIT_INTERRUPTED = 종료 신호로 중단, 1 = I/O 오류
int run_mcts_pipeline(const MCTSConfig& cfg)
{
    const bool has_attribute_damage = std::any_of(cfg.goal.begin(), cfg.goal.end(),
        [](const GoalEntry& entry) { return entry.opt_idx == 11; });
    auto inits = generate_inits(cfg.init_biscuit, cfg.id_start, cfg.id_end,
        has_attribute_damage, cfg.seed);

    const int total = cfg.id_end - cfg.id_start;

    // 서로 다른 수명/접근 빈도를 가진 원자 변수의 false sharing 방지.
    alignas(64) std::atomic<int> done{ 0 };
    alignas(64) std::atomic<int> next_id{ cfg.id_start };
    alignas(64) std::atomic<bool> producer_finished{ false };
    alignas(64) std::atomic<bool> io_failed{ false };

    // The 65,536-cell queue occupies about 8 MiB; keep it off the thread stack.
    auto queue = std::make_unique<LockFreeMPSCQueue<Record, 65536>>();

    auto t0 = std::chrono::steady_clock::now();

    std::thread io_thread([&]() {
        std::ofstream f(cfg.output_path, std::ios::app);
        if (!f.is_open()) {
            std::cerr << "[Error] 출력 파일 오픈 실패: " << cfg.output_path << std::endl;
            io_failed.store(true, std::memory_order_release);
            return;
        }

        std::vector<Record> write_buffer;
        write_buffer.reserve(1000);

        auto flush_buffer = [&](const std::vector<Record>& buf) {
            if (buf.empty()) return;
            for (const auto& r : buf) {
                for (int t : r.tier_counts) f << t << ",";
                f << r.progress << "," << r.bg << "," << r.bl;
                for (int t : r.opt_max_tier) f << "," << t;
                f << "," << r.best_action << "," << r.best_q
                  << "," << r.q_general << "," << r.q_light_avg
                  << "," << r.goal_id << "," << r.resource_tier
                  << "," << r.biscuit_id << "," << r.matched_at_switch
                  << "," << r.bl_remaining_at_switch
                  << "," << r.q_stop << "," << r.q_continue << "," << r.score
                  << "," << r.matched_count << "," << r.goal_achieved
                  << "," << r.bg_spent << "," << r.bl_spent
                  << "," << r.decision_step << "," << r.terminal_reason
                  << "," << r.rng_seed << "," << r.state_samples
                  << "," << r.state_progress_mean << "," << r.state_matched_mean
                  << "," << r.state_bg_mean << "," << r.state_bl_mean
                  << "," << r.state_score_mean << "," << r.state_matched_min
                  << "," << r.state_matched_max << "\n";
            }
            f.flush();
        };

        auto last_flush = std::chrono::steady_clock::now();
        Record rec;
        while (true) {
            if (queue->pop(rec)) {
                write_buffer.push_back(rec);
                if (write_buffer.size() >= 1000) {
                    flush_buffer(write_buffer);
                    write_buffer.clear();
                    last_flush = std::chrono::steady_clock::now();
                }
            } else {
                if (producer_finished.load(std::memory_order_acquire)) {
                    if (!queue->pop(rec)) {
                        break;
                    }
                    write_buffer.push_back(rec);
                } else {
                    // 큐가 비었을 때 busy-yield 대신 sleep: 워커에게 코어를 양보한다.
                    auto now = std::chrono::steady_clock::now();
                    if (!write_buffer.empty() &&
                        now - last_flush >= std::chrono::seconds(10)) {
                        flush_buffer(write_buffer);
                        write_buffer.clear();
                        last_flush = now;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        }
        flush_buffer(write_buffer);
    });

    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const int progress_step = std::max(1, total / 20);
    std::vector<std::thread> workers;

    for (unsigned int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            std::vector<Node> local_node_pool;
            // run(n_sim)은 simulation당 최대 1개 노드만 확장한다.
            // 따라서 트리의 최대 노드 수는 root 1개 + n_sim개다.
            local_node_pool.reserve(static_cast<size_t>(cfg.n_sim) + 1);

            while (true) {
                if (io_failed.load(std::memory_order_acquire)) break;
                // 시간제한/종료 신호: 새 비스킷을 시작하지 않는다 (진행 중인 비스킷은 끝까지 기록)
                if (g_stop.load(std::memory_order_relaxed)) break;

                int i = next_id.fetch_add(1, std::memory_order_relaxed);
                if (i >= cfg.id_end) break;

                local_node_pool.clear();
                const uint64_t biscuit_seed = mix_seed(cfg.seed ^ static_cast<uint64_t>(i) ^ 0x534541524348ULL);
                tl_rng.reseed(biscuit_seed);
                MCTSTree tree(local_node_pool, inits[static_cast<size_t>(i - cfg.id_start)],
                    cfg.budget_g, cfg.budget_l, cfg.goal, cfg.max_rollout);
                tree.run(cfg.n_sim);

                auto recs = tree.extract_records(200);
                if (!recs.empty()) {
                    for (auto& r : recs) {
                        r.goal_id = cfg.goal_id;
                        r.resource_tier = cfg.resource_tier;
                        r.biscuit_id = i;
                        r.rng_seed = mix_seed(cfg.seed ^ static_cast<uint64_t>(i) ^ 0x534541524348ULL);

                        while (!queue->push(r)) {
                            if (io_failed.load(std::memory_order_acquire)) return;
                            std::this_thread::yield();
                        }
                    }
                }

                int cnt = ++done;
                if (cnt % progress_step == 0) {
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::cout << "  " << cnt << "/" << total
                              << " (" << static_cast<long long>(el) << "s)" << std::endl;
                }
            }
        });
    }

    for (auto& w : workers) if (w.joinable()) w.join();
    producer_finished.store(true, std::memory_order_release);

    if (io_thread.joinable()) io_thread.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[MCTS] 탐색 종료: " << elapsed << "초 (" << done.load() << "/" << total << ")\n";

    if (io_failed.load()) return 1;
    return (done.load() >= total) ? 0 : EXIT_INTERRUPTED;
}

static void print_usage() {
    std::cerr <<
        "사용법: mcts_engine [옵션]\n"
        "  --tier T            자원 티어 0~4 (생략 시 전체)\n"
        "  --goal G            목표 0~9    (생략 시 전체)\n"
        "  --id_start S        biscuit_id 시작 (기본 0)\n"
        "  --id_end E          biscuit_id 끝, 미포함 (기본 n_starts)\n"
        "  --out PATH          출력 CSV (--tier/--goal 둘 다 지정 시에만 사용)\n"
        "  --n_starts N        기본 100000\n"
        "  --n_sim N           기본 500\n"
        "  --seed N            재현 가능한 64-bit seed\n";
}

int main(int argc, char* argv[]) {
    build_opt_vals();
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::vector<Goal> GOALS = {
        { { 5,4,5 } },
        { { 5,4,4 } },
        { { 10,4,5 } },
        { { 10,4,4 } },
        { { 3,4,5 } },
        { { 3,4,4 } },
        { { 11,4,5 } },
        { { 11,4,4 } },
        { { 4,2,5 },{ 3,2,5 } },
        { { 4,2,4 },{ 3,2,4 } },
    };

    struct Tier { int bg; int bl; int id; };
    const std::vector<Tier> TIERS = {
        { 3200,   23,  0 },
        { 6666,   44,  1 },
        { 13333,  76,  2 },
        { 43333,  196, 3 },
        { 103333, 388, 4 },
    };

    // ── CLI 파싱 ──
    int tier_filter = -1, goal_filter = -1;
    int id_start = 0, id_end = -1;
    int n_starts = 100000, n_sim = 500;
    uint64_t base_seed = 20261123ULL;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "[Error] 값이 필요합니다: " << a << "\n";
                print_usage();
                std::exit(2);
            }
            return argv[++i];
        };
        try {
            if (a == "--tier")                tier_filter = std::stoi(need());
            else if (a == "--goal")           goal_filter = std::stoi(need());
            else if (a == "--id_start")       id_start = std::stoi(need());
            else if (a == "--id_end")         id_end = std::stoi(need());
            else if (a == "--out")            out_path = need();
            else if (a == "--n_starts")       n_starts = std::stoi(need());
            else if (a == "--n_sim")          n_sim = std::stoi(need());
            else if (a == "--seed")            base_seed = std::stoull(need());
            else { std::cerr << "[Error] 알 수 없는 인자: " << a << "\n"; print_usage(); return 2; }
        } catch (const std::exception&) {
            std::cerr << "[Error] 잘못된 값: " << a << "\n";
            return 2;
        }
    }

    if (id_end < 0 || id_end > n_starts) id_end = n_starts;
    if (id_start < 0 || id_start >= id_end) {
        std::cerr << "[Error] 잘못된 id 범위: [" << id_start << ", " << id_end << ")\n";
        return 2;
    }
    if (tier_filter >= static_cast<int>(TIERS.size()) || goal_filter >= static_cast<int>(GOALS.size())) {
        std::cerr << "[Error] tier/goal 범위 초과\n";
        return 2;
    }
    if (!out_path.empty() && (tier_filter < 0 || goal_filter < 0)) {
        std::cerr << "[Error] --out은 --tier와 --goal을 함께 지정해야 합니다.\n";
        return 2;
    }

    std::cout << "[Info] threads=" << std::max(1u, std::thread::hardware_concurrency())
              << " id_range=[" << id_start << "," << id_end << ")"
              << " n_sim=" << n_sim << std::endl;

    for (int ti = 0; ti < (int)TIERS.size(); ++ti) {
        if (tier_filter >= 0 && ti != tier_filter) continue;
        for (int gi = 0; gi < (int)GOALS.size(); ++gi) {
            if (goal_filter >= 0 && gi != goal_filter) continue;

            std::string filename = !out_path.empty() ? out_path
                : "biscuit_tier_" + std::to_string(ti) + "_goal_" + std::to_string(gi) + ".csv";

            MCTSConfig cfg;
            cfg.goal = GOALS[gi];
            cfg.goal_id = gi;
            cfg.resource_tier = ti;
            cfg.budget_g = TIERS[ti].bg;
            cfg.budget_l = TIERS[ti].bl;
            cfg.max_rollout = TIERS[ti].bg + TIERS[ti].bl;
            cfg.n_starts = n_starts;
            cfg.n_sim = n_sim;
            cfg.seed = base_seed;
            cfg.id_start = id_start;
            cfg.id_end = id_end;
            cfg.output_path = filename;

            auto parent = std::filesystem::path(filename).parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
            }

            {
                std::ofstream hdr(filename, std::ios::trunc);
                if (!hdr) {
                    std::cerr << "[Error] 출력 파일 생성 실패: " << filename << "\n";
                    return 1;
                }
                hdr << CSV_HEADER;
            }
            std::cout << "\n[New ] " << filename << " 생성\n";

            std::cout << "[Tier " << ti << " Goal " << gi << " 시작]" << std::endl;
            int status = run_mcts_pipeline(cfg);
            if (status != 0) {
                if (status == EXIT_INTERRUPTED)
                    std::cout << "[Stop] 종료 신호 수신 → 안전 종료 (exit " << status << ")\n";
                return status;
            }
        }
    }

    std::cout << "\n[All Done] 모든 청크 완료!\n";
    return 0;
}
