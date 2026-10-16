/**
 * mcts_engine.cpp — 비스킷 리롤 MCTS 엔진 (최적화 완료 버전)
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <array>

 // ── 옵션 ──────────────────────────────────────────────────
struct Option { std::string name; float lo, hi; };
static const std::vector<Option> OPTIONS = {
    { "공격력",          3.0f, 7.5f },
    { "방어력",          5.0f, 7.5f },
    { "체력",            3.0f, 15.0f },
    { "공격속도",        3.0f, 10.0f },
    { "치명타 확률",     3.0f, 7.0f },
    { "쿨타임",          2.0f, 6.0f },
    { "피해감소",        5.0f, 10.0f },
    { "치명타 피해감소", 4.0f, 10.0f },
    { "이로운 효과 증가",2.0f, 5.0f },
    { "해로운 효과 감소",2.0f, 5.0f },
    { "피해감소 무시",   5.0f, 15.0f },
    { "속성 피해 증가",  8.0f, 15.0f },
};
static const int N_OPTS = 12;
static const std::vector<float> WEIGHTS = {
    1.f,1.f,1.f,5.f,1.f,5.f,5.f,1.f,1.f,1.f,5.f,5.f
};
static std::vector<std::vector<float>> OPT_VALS;
static std::vector<int> OPT_SIZES;

void build_opt_vals() {
    OPT_VALS.resize(N_OPTS);
    OPT_SIZES.resize(N_OPTS);
    for (int i = 0; i < N_OPTS; ++i) {
        float lo = OPTIONS[i].lo, hi = OPTIONS[i].hi;
        int steps = static_cast<int>(std::round((hi - lo) / 0.1f)) + 1;
        OPT_VALS[i].resize(static_cast<size_t>(steps));
        OPT_SIZES[i] = steps;
        for (int j = 0; j < steps; ++j)
            OPT_VALS[i][static_cast<size_t>(j)] =
            std::round((lo + static_cast<float>(j) * 0.1f) * 10.f) / 10.f;
    }
}

// ── FastRNG (Fast Reduction 적용 / Boundary Safe) ──────────────────
struct FastRNG {
    using result_type = uint64_t;
    uint64_t s[2];
    explicit FastRNG(uint64_t seed) {
        s[0] = seed ^ 0x6C0789651D230094ULL;
        s[1] = (seed * 13137ULL) ^ 0x9E3779B97F4A7C15ULL;
    }
    inline result_type operator()() {
        uint64_t s1 = s[0]; const uint64_t s0 = s[1];
        s[0] = s0; s1 ^= s1 << 23;
        s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        return s[1] + s0;
    }
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return UINT64_MAX; }

    // Fast Reduction (32비트 고속 매핑)
    inline int next_int(int n) {
        if (n <= 1) return 0;
        uint32_t r = static_cast<uint32_t>(operator()());
        return static_cast<int>((static_cast<uint64_t>(r) * static_cast<uint64_t>(n)) >> 32);
    }
};
thread_local FastRNG tl_rng{ std::random_device{}() };

// ── 슬롯 / 비스킷 / 목표 ──────────────────────────────────
struct Slot { int opt_idx; float val; };
using Biscuit = std::array<Slot, 4>;

struct GoalEntry { int opt_idx, required, min_tier; };
using Goal = std::vector<GoalEntry>;

// ── [완전 교정] get_tier (Max 옵션 무조건 주황 + 공식 등급 컷: 1% / 10% / 25% / 50%) ──
inline int get_tier(int opt_idx, float val) {
    float lo = OPTIONS[static_cast<size_t>(opt_idx)].lo;
    float hi = OPTIONS[static_cast<size_t>(opt_idx)].hi;
    if (hi <= lo) return 0;

    // 🌟 [핵심] 맥옵(Max)이면 부동소수점 오차 불문하고 무조건 5등급(주황) 하이패스!
    if (val >= hi - 1e-5f) return 5;

    // ratio: 옵션 최저점(0.0) ~ 최고점(1.0) 비율
    float ratio = (val - lo) / (hi - lo);

    // 상위 백분율 기준 매핑
    if (ratio >= 0.99f) return 5; // 🟧 주황 (상위 1% 이내)
    if (ratio >= 0.90f) return 4; // 🟪 보라 (상위 10% 이내, ratio 0.90 이상)
    if (ratio >= 0.75f) return 3; // 🟦 파랑 (상위 25% 이내)
    if (ratio >= 0.50f) return 2; // 🟩 초록 (상위 50% 이내)
    return 1;                     // ⬜ 하양 (하위 50%)
}

// ── [최적화] 1:1 슬롯 매칭 (힙 할당 0Byte, Inline) ──
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

float slot_contribution(int opt_idx, float val) {
    float lo = OPTIONS[static_cast<size_t>(opt_idx)].lo;
    float hi = OPTIONS[static_cast<size_t>(opt_idx)].hi;
    float norm = (hi > lo) ? (val - lo) / (hi - lo) : 1.0f;
    return WEIGHTS[static_cast<size_t>(opt_idx)] * norm * 10.0f;
}

float biscuit_score(const Biscuit& b) {
    float t = 0.0f;
    for (const auto& s : b) t += slot_contribution(s.opt_idx, s.val);
    return t;
}

inline Slot sample_slot() {
    int opt = tl_rng.next_int(N_OPTS);
    int idx = tl_rng.next_int(OPT_SIZES[static_cast<size_t>(opt)]);
    return { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(idx)] };
}

Biscuit reroll_general(const Biscuit&) {
    Biscuit nb;
    for (auto& s : nb) s = sample_slot();
    return nb;
}

// 빛나는 도우 리롤 (유저 직접 고유 판단 공간 확보)
Biscuit reroll_light(const Biscuit& b, int pos) {
    Biscuit nb = b;
    nb[static_cast<size_t>(pos)] = sample_slot();
    return nb;
}

// [개선] Reward 정규화 (0.0 ~ 1.0) -> UCB1 정상 작동 보장
// ── [수정 후] compute_reward ──
// ── [수정] compute_reward ──
// ── [보상 함수 전면 개편] ──
float compute_reward(const Biscuit& b, const Goal& goal, bool success) {
    // 1. 목표 4개 완벽 달성 시 압도적인 대박 보상 부여
    if (success) {
        return 10.0f;
    }

    float prog = goal_progress(b, goal); // 0.0(0개), 0.25(1개), 0.5(2개), 0.75(3개)
    float score_norm = biscuit_score(b) / 200.0f;

    // 2. 달성도(prog)에 따른 기하급수적 보상 배율 설정
    // 0개: 0.0
    // 1개(0.25): 0.125
    // 2개(0.50): 1.000
    // 3개(0.75): 3.375  <-- 3개 맞춘 상태의 가치를 0.5일 때보다 3배 이상 폭등시킴!
    float prog_weight = std::pow(prog * 2.0f, 3.0f);

    // 3. 자원 아낀 보너스 (빛도우를 안 쓰고 보라 3개를 유지하면 추가 가산점)
    return prog_weight + score_norm * 0.1f;
}

// ── MCTS ──────────────────────────────────────────────────
static const int N_ACTIONS = 5;

// ── [근본 수정] Node 구조체 슬롯 초기화 ──
struct Node {
    Biscuit biscuit{}; // 💡 {}를 붙여서 0으로 기본 초기화!
    int bg = 0, bl = 0, parent = -1, action = -1;
    std::array<int, N_ACTIONS> children{ -1,-1,-1,-1,-1 };
    std::array<bool, N_ACTIONS> untried{ true,true,true,true,true };
    float value = 0.0f;
    int visits = 0;
    bool is_terminal() const { return bg < 1 && bl < 1; }
    bool can_act(int a) const { return a == 0 ? bg >= 1 : bl >= 1; }
};

struct Record {
    std::array<int, 5>  tier_counts{};
    std::array<int, 12> opt_max_tier{};
    float progress = 0.0f, best_q = 0.0f, q_general = 0.0f, q_light_avg = 0.0f;
    int bg = 0, bl = 0, best_action = -1;

    int goal_id = -1;
    int resource_tier = -1;  // 0=50%, 1=25%, 2=10%, 3=1%, 4=0.1%
    int biscuit_id = -1;
    int matched_at_switch = 0;   // Q1: 빛도우 진입 시 맞춘 슬롯 수
    int bl_remaining_at_switch = 0; // Q2: 빛도우 진입 시 잔여 BL
};

struct MCTSTree {
    std::vector<Node>& nodes;
    const Goal& goal;
    int max_rollout;

    MCTSTree(std::vector<Node>& pool, const Biscuit& init,
        int bg, int bl, const Goal& g, int mr)
        : nodes(pool), goal(g), max_rollout(mr)
    {
        Node root;
        root.biscuit = init; root.bg = bg; root.bl = bl;
        nodes.push_back(root);
    }

    int new_node(const Biscuit& b, int bg, int bl, int par, int act) {
        Node n;
        n.biscuit = b; n.bg = bg; n.bl = bl;
        n.parent = par; n.action = act;
        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
    }

    // [개선] log_parent 미리 전달 받아 UCB1 연산 최소화 (상수 C=0.7f로 스케일 최적화)
    float ucb1(int ni, int a, float log_parent_visits, float c = 0.707f) {
        int ci = nodes[static_cast<size_t>(ni)].children[static_cast<size_t>(a)];
        if (ci < 0) return std::numeric_limits<float>::infinity();
        const Node& child = nodes[static_cast<size_t>(ci)];
        if (child.visits == 0) return std::numeric_limits<float>::infinity();
        return (child.value / static_cast<float>(child.visits)) +
            c * std::sqrt(log_parent_visits / static_cast<float>(child.visits));
    }

    int select() {
        int idx = 0;
        while (!nodes[static_cast<size_t>(idx)].is_terminal()) {
            const Node& n = nodes[static_cast<size_t>(idx)];
            bool has_untried = false;
            for (int a = 0; a < N_ACTIONS; ++a)
                if (n.untried[static_cast<size_t>(a)] && n.can_act(a)) {
                    has_untried = true; break;
                }
            if (has_untried) return idx;

            int best_a = -1; float best_u = -1.0f;
            float log_p = std::log(static_cast<float>(n.visits)); // 1회만 계산

            for (int a = 0; a < N_ACTIONS; ++a) {
                if (!n.can_act(a)) continue;
                float u = ucb1(idx, a, log_p);
                if (u > best_u) { best_u = u; best_a = a; }
            }
            if (best_a < 0 || n.children[static_cast<size_t>(best_a)] < 0) return idx;
            idx = n.children[static_cast<size_t>(best_a)];
        }
        return idx;
    }

    // ── [수정 후] MCTSTree::expand ──
    int expand(int idx) {
        Node& node = nodes[static_cast<size_t>(idx)];

        // 목표 달성 슬롯이 하나도 없으면 빛도우 봉인
        bool has_goal_slot = false;
        for (const auto& s : node.biscuit) {
            for (const auto& g : goal) {
                if (s.opt_idx == g.opt_idx &&
                    get_tier(s.opt_idx, s.val) >= g.min_tier) {
                    has_goal_slot = true;
                    break;
                }
            }
            if (has_goal_slot) break;
        }

        std::array<int, N_ACTIONS> acts{}; int n_acts = 0;
        for (int a = 0; a < N_ACTIONS; ++a) {
            if (!node.untried[static_cast<size_t>(a)] || !node.can_act(a)) continue;

            // 목표 달성 슬롯 없으면 빛도우 봉인
            if (a >= 1 && !has_goal_slot) continue;

            // 목표 달성된 슬롯은 빛도우 대상에서 제외
            if (a >= 1) {
                int pos = a - 1;
                bool is_goal_slot = false;
                for (const auto& g : goal) {
                    if (node.biscuit[static_cast<size_t>(pos)].opt_idx == g.opt_idx &&
                        get_tier(node.biscuit[static_cast<size_t>(pos)].opt_idx,
                            node.biscuit[static_cast<size_t>(pos)].val) >= g.min_tier) {
                        is_goal_slot = true;
                        break;
                    }
                }
                if (is_goal_slot) continue;
            }

            acts[static_cast<size_t>(n_acts++)] = a;
        }
        if (n_acts == 0) return idx;

        int a = acts[static_cast<size_t>(tl_rng.next_int(n_acts))];
        node.untried[static_cast<size_t>(a)] = false;

        Biscuit nb; int new_bg = node.bg, new_bl = node.bl;
        if (a == 0) { nb = reroll_general(node.biscuit); --new_bg; }
        else { nb = reroll_light(node.biscuit, a - 1); --new_bl; }

        int ci = new_node(nb, new_bg, new_bl, idx, a);
        nodes[static_cast<size_t>(idx)].children[static_cast<size_t>(a)] = ci;
        return ci;
    }

    // ── [수정 후] MCTSTree::rollout ──
    // ── [완전 수정] MCTSTree::rollout ──
    // ── [최적화] MCTSTree::rollout (Zero-Allocation 스나이핑) ──
    float rollout(int idx) {
        Biscuit b = nodes[static_cast<size_t>(idx)].biscuit;
        int bg = nodes[static_cast<size_t>(idx)].bg;
        int bl = nodes[static_cast<size_t>(idx)].bl;
        bool success = false;

        for (int step = 0; step < max_rollout; ++step) {
            if (check_goal(b, goal)) { success = true; break; }

            int a = -1;
            auto matched = get_matched_slots(b, goal);

            // 💡 std::vector 대신 스택 배열(std::array)을 사용해 메모리 할당 방지
            std::array<int, 4> unmatched_actions{};
            int unmatched_cnt = 0;

            for (int i = 0; i < 4; ++i) {
                if (!matched[i]) {
                    unmatched_actions[unmatched_cnt++] = i + 1; // 슬롯 0~3 -> 액션 1~4
                }
            }

            float prog = static_cast<float>(4 - unmatched_cnt) / 4.0f; // 빠르게 progress 계산

            // 🎯 스나이핑 로직
            if (bl >= 1 && unmatched_cnt > 0 && (prog >= 0.5f || bg < 1)) {
                a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
            }
            else if (bg >= 1 && bl >= 1) {
                if (tl_rng.next_int(10) < 8) {
                    a = 0;
                }
                else if (unmatched_cnt > 0) {
                    a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
                }
                else {
                    a = 0;
                }
            }
            else if (bg >= 1) {
                a = 0;
            }
            else if (bl >= 1 && unmatched_cnt > 0) {
                a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
            }
            else break;

            if (a == 0) { b = reroll_general(b); --bg; }
            else { b = reroll_light(b, a - 1); --bl; }
        }
        return compute_reward(b, goal, success);
    }

    void backprop(int idx, float r) {
        while (idx >= 0) {
            Node& n = nodes[static_cast<size_t>(idx)];
            ++n.visits; n.value += r;
            idx = n.parent;
        }
    }

    void run(int n_sim) {
        for (int i = 0; i < n_sim; ++i) {
            int leaf = select();
            if (!nodes[static_cast<size_t>(leaf)].is_terminal())
                leaf = expand(leaf);
            backprop(leaf, rollout(leaf));
        }
    }

    std::vector<Record> extract_records(int max_depth = 200) {
        std::vector<Record> recs;
        int idx = 0;
        for (int depth = 0; depth < max_depth; ++depth) {
            const Node& node = nodes[static_cast<size_t>(idx)];
            if (node.is_terminal()) break;

            int best_a = -1; float best_q = -1e9f;
            float q_gen = 0.0f, q_light_sum = 0.0f; int q_light_cnt = 0;

            for (int a = 0; a < N_ACTIONS; ++a) {
                int ci = node.children[static_cast<size_t>(a)];
                if (ci < 0) continue;
                const Node& child = nodes[static_cast<size_t>(ci)];
                if (child.visits == 0) continue;
                float q = child.value / static_cast<float>(child.visits);
                if (q > best_q) { best_q = q; best_a = a; }
                if (a == 0) q_gen = q;
                else { q_light_sum += q; ++q_light_cnt; }
            }
            if (best_a < 0) break;

            Record rec;
            rec.progress = goal_progress(node.biscuit, goal);
            rec.bg = node.bg;
            rec.bl = node.bl;
            rec.best_action = best_a;
            rec.best_q = best_q;
            rec.q_general = q_gen;
            rec.q_light_avg = q_light_cnt > 0 ?
                q_light_sum / static_cast<float>(q_light_cnt) : 0.0f;
            rec.tier_counts.fill(0);
            rec.opt_max_tier.fill(0);
            for (const auto& s : node.biscuit) {
                int t = get_tier(s.opt_idx, s.val);
                rec.tier_counts[static_cast<size_t>(t)]++;
                int& mx = rec.opt_max_tier[static_cast<size_t>(s.opt_idx)];
                if (t > mx) mx = t;
            }
            recs.push_back(rec);

            int ci = node.children[static_cast<size_t>(best_a)];
            if (ci < 0) break;
            idx = ci;
        }
        return recs;
    }
};

// ── 병렬 탐색 ─────────────────────────────────────────────
struct TaskResult {
    std::vector<Record> records;
    bool valid = false;
};

TaskResult run_one(const Biscuit& init, int bg, int bl,
    const Goal& goal, int n_sim, int max_rollout)
{
    thread_local std::vector<Node> node_pool;
    node_pool.clear();
    if (node_pool.capacity() < 2000000) node_pool.reserve(2000000);

    MCTSTree tree(node_pool, init, bg, bl, goal, max_rollout);
    tree.run(n_sim);

    auto recs = tree.extract_records(200);
    if (recs.empty()) return { {}, false };
    return { std::move(recs), true };
}

// ── 초기 상태 생성 ─────────────────────────────────────────
/*std::vector<Biscuit> generate_inits(const Biscuit& base, int n, std::mt19937& rng) {
    std::vector<Biscuit> inits(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (i % 3 == 0) {
            for (int p = 0; p < 4; ++p) {
                int opt = static_cast<int>(rng() % static_cast<unsigned>(N_OPTS));
                int vidx = static_cast<int>(rng() % static_cast<unsigned>(OPT_SIZES[static_cast<size_t>(opt)]));
                inits[static_cast<size_t>(i)][static_cast<size_t>(p)] =
                { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(vidx)] };
            }
        }
        else {
            inits[static_cast<size_t>(i)] = base;
            int nr = static_cast<int>(rng() % 3u);
            for (int j = 0; j < nr; ++j) {
                int pos = static_cast<int>(rng() % 4u);
                int opt = static_cast<int>(rng() % static_cast<unsigned>(N_OPTS));
                int vidx = static_cast<int>(rng() % static_cast<unsigned>(OPT_SIZES[static_cast<size_t>(opt)]));
                inits[static_cast<size_t>(i)][static_cast<size_t>(pos)] =
                { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(vidx)] };
            }
        }
    }
    return inits;
}*/

// ── [근본 수정] generate_inits ──
std::vector<Biscuit> generate_inits(const Biscuit& base, int n, std::mt19937& rng) {
    std::vector<Biscuit> inits(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (i % 3 == 0) {
            for (int p = 0; p < 4; ++p) {
                // 💡 rng() % N_OPTS 대신 tl_rng.next_int(N_OPTS) 사용!
                int opt = tl_rng.next_int(N_OPTS);
                int vidx = tl_rng.next_int(OPT_SIZES[static_cast<size_t>(opt)]);
                inits[static_cast<size_t>(i)][static_cast<size_t>(p)] =
                { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(vidx)] };
            }
        }
        else {
            inits[static_cast<size_t>(i)] = base;
            int nr = tl_rng.next_int(3); // 0, 1, 2
            for (int j = 0; j < nr; ++j) {
                int pos = tl_rng.next_int(4);
                int opt = tl_rng.next_int(N_OPTS);
                int vidx = tl_rng.next_int(OPT_SIZES[static_cast<size_t>(opt)]);
                inits[static_cast<size_t>(i)][static_cast<size_t>(pos)] =
                { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(vidx)] };
            }
        }
    }
    return inits;
}

// ── CSV 출력 ──────────────────────────────────────────────
void write_csv(const std::vector<Record>& records, const std::string& path) {
    std::ofstream f(path);
    f << "tier_0,tier_1,tier_2,tier_3,tier_4,progress,budget_g,budget_l";
    for (int i = 0; i < 12; ++i) f << ",opt" << i << "_tier";
    f << ",best_action,best_q,q_general,q_light_avg,goal_id,resource_tier,biscuit_id,matched_at_switch,bl_remaining_at_switch\n";
    for (const auto& r : records) {
        for (int t : r.tier_counts) f << t << ",";
        f << r.progress << "," << r.bg << "," << r.bl;
        for (int t : r.opt_max_tier) f << "," << t;
        f << "," << r.best_action << "," << r.best_q
            << "," << r.q_general << "," << r.q_light_avg
            << "," << r.goal_id << "," << r.resource_tier
            << "," << r.biscuit_id << "," << r.matched_at_switch
            << "," << r.bl_remaining_at_switch << "\n";
    }
    std::cout << "CSV 저장: " << path << " (" << records.size() << "개)\n";
}

// ── 메인 ──────────────────────────────────────────────────
// (기존 struct Slot, Biscuit, Goal, TaskResult, Record, run_one, generate_inits, write_csv 등은 상단 헤더/선언부 포함)

// ── 실행 설정 구조체 ───────────────────────────────────────
/**
 * MCTSConfig 및 완벽한 CLI 파서 (초기 비스킷 + 목표 + 자원 + 파라미터)
 */
 // ── [근본 해결] MCTSConfig 정의부에 init_biscuit 기본값 장착 ──
struct MCTSConfig {
    // 💡 디폴트 생성 시에도 유효한 인덱스(0~11 범위 내)를 갖도록 기본값 명시!
    Biscuit init_biscuit = { Slot{ 0, 3.0f }, Slot{ 1, 5.0f }, Slot{ 3, 3.0f }, Slot{ 4, 3.0f } };
    Goal goal = { { 3, 2, 3 },{ 4, 2, 3 } };
    int budget_g = 90000;
    int budget_l = 3000;
    int n_starts = 100000;
    int n_sim = 2000;
    int max_rollout = 300;
    int goal_id = 0;
    int resource_tier = 0;
    std::string output_path = "";
};

MCTSConfig parse_args(int argc, char* argv[]) {
    MCTSConfig cfg;

    // 1. 기본값 설정 (인자가 안 들어왔을 때의 Fallback)
    cfg.init_biscuit = { {
        { 0, 3.0f },{ 1, 5.0f },{ 8, 2.0f },{ 9, 2.0f }
    } };
    cfg.goal = { { 3, 2, 3 },{ 4, 2, 3 } };

    // 2. CLI 인자 루프 파싱
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--bg" && i + 1 < argc) {
            cfg.budget_g = std::stoi(argv[++i]);
        }
        else if (arg == "--bl" && i + 1 < argc) {
            cfg.budget_l = std::stoi(argv[++i]);
        }
        else if (arg == "--starts" && i + 1 < argc) {
            cfg.n_starts = std::stoi(argv[++i]);
        }
        else if (arg == "--sim" && i + 1 < argc) {
            cfg.n_sim = std::stoi(argv[++i]);
        }
        else if (arg == "--rollout" && i + 1 < argc) {
            cfg.max_rollout = std::stoi(argv[++i]);
        }
        else if (arg == "--out" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        }

        // 초기 비스킷 슬롯 파싱 (예: --slots "0:3.0,1:5.0,8:2.0,9:2.0")
        else if (arg == "--slots" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string item;
            int idx = 0;
            while (std::getline(ss, item, ',') && idx < 4) {
                auto colon = item.find(':');
                if (colon != std::string::npos) {
                    int opt = std::stoi(item.substr(0, colon));
                    float val = std::stof(item.substr(colon + 1));
                    cfg.init_biscuit[static_cast<size_t>(idx++)] = { opt, val };
                }
            }
        }

        // 🎯 목표 조건 파싱 (예: --goal "3:2:3,4:2:3")
        else if (arg == "--goal" && i + 1 < argc) {
            cfg.goal.clear(); // 기본 목표 지우고 새로 채움
            std::stringstream ss(argv[++i]);
            std::string item;

            while (std::getline(ss, item, ',')) {
                std::stringstream entry_ss(item);
                std::string token;
                std::vector<int> tokens;

                while (std::getline(entry_ss, token, ':')) {
                    tokens.push_back(std::stoi(token));
                }

                // [옵션ID, 필요개수, 최소등급] 3개 세트 검증
                if (tokens.size() == 3) {
                    cfg.goal.push_back({ tokens[0], tokens[1], tokens[2] });
                }
            }
        }
    }

    return cfg;
}

// ── MCTS 대규모 탐색 코어 엔진 ──────────────────────────────
//std::vector<Record> run_mcts_pipeline(const MCTSConfig& cfg) {
//    std::mt19937 rng{ 42u };
//    auto inits = generate_inits(cfg.init_biscuit, cfg.n_starts, rng);
//
//    std::vector<TaskResult> task_results(static_cast<size_t>(cfg.n_starts));
//    std::atomic<int> done{ 0 };
//
//    auto t0 = std::chrono::steady_clock::now();
//
//    std::transform(
//        std::execution::par,
//        inits.begin(), inits.end(),
//        task_results.begin(),
//        [&](const Biscuit& b) -> TaskResult {
//            auto r = run_one(b, cfg.budget_g, cfg.budget_l, cfg.goal, cfg.n_sim, cfg.max_rollout);
//            int cnt = ++done;
//            if (cnt % 100 == 0)
//                std::cout << "  " << cnt << "/" << cfg.n_starts << "\n";
//            return r;
//        }
//    );
//
//    double elapsed = std::chrono::duration<double>(
//        std::chrono::steady_clock::now() - t0).count();
//
//    std::cout << "\n[MCTS] 탐색 완료: " << elapsed << "초\n";
//
//    std::vector<Record> all_records;
//    int bid = 0;
//    for (auto& tr : task_results) {
//        if (tr.valid) {
//            for (auto& r : tr.records) {
//                r.goal_id = cfg.goal_id;
//                r.resource_tier = cfg.resource_tier;
//                r.biscuit_id = bid;
//                all_records.push_back(std::move(r));
//            }
//            ++bid;
//        }
//    }
//
//    return all_records;
//}

// ── run_mcts_pipeline 함수를 아래 구조로 교체 ──────────────────────────────
// ── [캐글 CPU 환경 맞춤형] run_mcts_pipeline ──────────────────────────────
std::vector<Record> run_mcts_pipeline(const MCTSConfig& cfg) {
    std::mt19937 rng{ 42u };
    auto inits = generate_inits(cfg.init_biscuit, cfg.n_starts, rng);

    std::vector<TaskResult> task_results(static_cast<size_t>(cfg.n_starts));
    std::atomic<int> done{ 0 };

    auto t0 = std::chrono::steady_clock::now();

    // 🎯 [캐글 최적화 핵심] vCPU 4개에 속지 않고, 물리 코어 개수인 '2개'로 스레드 강제 고정!
    // 하이퍼스레딩 경합을 없애고 CPU 연산 효율을 극대화합니다.
    const unsigned int num_threads = 2;

    std::vector<std::thread> workers;
    int chunk_size = cfg.n_starts / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_idx = t * chunk_size;
        int end_idx = (t == num_threads - 1) ? cfg.n_starts : (t + 1) * chunk_size;

        workers.emplace_back([&, start_idx, end_idx]() {
            // 💡 딱 2개의 스레드만 이 거대한 메모리 풀을 각각 1번씩만 할당받아 '완벽하게 무한 재사용'합니다.
            // 메모리 할당/해제 오버헤드가 제로(0)가 됩니다.
            std::vector<Node> local_node_pool;
            local_node_pool.reserve(2000000);

            for (int i = start_idx; i < end_idx; ++i) {
                local_node_pool.clear();
                MCTSTree tree(local_node_pool, inits[i], cfg.budget_g, cfg.budget_l, cfg.goal, cfg.max_rollout);
                tree.run(cfg.n_sim);

                auto recs = tree.extract_records(200);
                if (!recs.empty()) {
                    task_results[i] = { std::move(recs), true };
                }

                int cnt = ++done;
                // 캐글 로그 스팸 방지 및 출력 락 오버헤드 최소화를 위해 출력 단위를 10,000으로 확장
                if (cnt % 10000 == 0) {
                    std::cout << "  " << cnt << "/" << cfg.n_starts << std::endl;
                }
            }
            });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n[MCTS] 탐색 완료: " << elapsed << "초\n";

    std::vector<Record> all_records;
    int bid = 0;
    for (auto& tr : task_results) {
        if (tr.valid) {
            for (auto& r : tr.records) {
                r.goal_id = cfg.goal_id;
                r.resource_tier = cfg.resource_tier;
                r.biscuit_id = bid;
                all_records.push_back(std::move(r));
            }
            ++bid;
        }
    }

    return all_records;
}



// ── 메인 함수 (극도의 깔끔함) ──────────────────────────────
// ── 메인 함수 (청크별 즉시 저장 + Resume 이어하기 로직 적용) ──────────────────────────────
int main(int argc, char* argv[]) {
    build_opt_vals();

    // 목표 10개 정의
    const std::vector<Goal> GOALS = {
        { { 5,4,5 } },           // 0: 쿨4×올주황
        { { 5,4,4 } },           // 1: 쿨4×올보라
        { { 10,4,5 } },          // 2: 피감무4×올주황
        { { 10,4,4 } },          // 3: 피감무4×올보라
        { { 3,4,5 } },           // 4: 공속4×올주황
        { { 3,4,4 } },           // 5: 공속4×올보라
        { { 11,4,5 } },          // 6: 속피증4×올주황
        { { 11,4,4 } },          // 7: 속피증4×올보라
        { { 4,2,5 },{ 3,2,5 } }, // 8: 치명2+공속2×올주황
        { { 4,2,4 },{ 3,2,4 } }, // 9: 치명2+공속2×올보라
    };

    struct Tier { int bg; int bl; int id; };
    const std::vector<Tier> TIERS = {
        { 3200,  23,  0 },  // 50%
        { 6666,  44,  1 },  // 25%
        { 13333, 76,  2 },  // 10%
        { 43333, 196, 3 },  // 1%
        { 103333,388, 4 },  // 0.1%
    };

    for (int ti = 0; ti < (int)TIERS.size(); ++ti) {
        for (int gi = 0; gi < (int)GOALS.size(); ++gi) {
            // 조합별 고유 청크 파일명 생성 (예: biscuit_tier_0_goal_3.csv)
            std::string filename = "biscuit_tier_" + std::to_string(ti) + "_goal_" + std::to_string(gi) + ".csv";

            // 🛡️ [Resume 이어하기 로직] 이미 해당 파일이 존재하면 연산을 통째로 건너뜀!
            std::ifstream check_file(filename);
            if (check_file.good()) {
                std::cout << "\n[Skip] 이미 완료된 청크 발견, 건너뜀: " << filename << "\n";
                check_file.close();
                continue;
            }
            check_file.close();

            std::cout << "\n[Tier " << ti << " Goal " << gi << " 시작]\n";

            MCTSConfig cfg;
            cfg.goal = GOALS[gi];
            cfg.goal_id = gi;
            cfg.resource_tier = ti;
            cfg.budget_g = TIERS[ti].bg;
            cfg.budget_l = TIERS[ti].bl;
            cfg.max_rollout = TIERS[ti].bg + TIERS[ti].bl;
            cfg.n_starts = 100000;
            cfg.n_sim = 500;
            cfg.output_path = filename;

            // MCTS 파이프라인 실행
            auto batch = run_mcts_pipeline(cfg);

            // 💡 결과를 메모리에 모아두지 않고, 조합이 끝나는 즉시 개별 CSV 파일로 저장!
            write_csv(batch, filename);
        }
    }

    std::cout << "\n[All Done] 모든 청크 데이터 수집 및 저장 완료!\n";
    return 0;
}
