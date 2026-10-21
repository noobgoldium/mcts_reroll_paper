/**
 * mcts_engine.cpp — 비스킷 리롤 MCTS 엔진 (락프리 링버퍼 + 완벽한 종료/드레이닝 마감 버전)
 */
#include <mutex>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <unordered_set>

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

// ── FastRNG ──────────────────────────────────────────────
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

inline int get_tier(int opt_idx, float val) {
    float lo = OPTIONS[static_cast<size_t>(opt_idx)].lo;
    float hi = OPTIONS[static_cast<size_t>(opt_idx)].hi;
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

Biscuit reroll_light(const Biscuit& b, int pos) {
    Biscuit nb = b;
    nb[static_cast<size_t>(pos)] = sample_slot();
    return nb;
}

float compute_reward(const Biscuit& b, const Goal& goal, bool success) {
    if (success) return 10.0f;
    float prog = goal_progress(b, goal);
    float score_norm = biscuit_score(b) / 200.0f;
    float prog_weight = std::pow(prog * 2.0f, 3.0f);
    return prog_weight + score_norm * 0.1f;
}

// ── MCTS ──────────────────────────────────────────────────
static const int N_ACTIONS = 5;

struct Node {
    Biscuit biscuit{};
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
    int resource_tier = -1;
    int biscuit_id = -1;
    int matched_at_switch = 0;
    int bl_remaining_at_switch = 0;
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
            float log_p = std::log(static_cast<float>(n.visits));
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

    int expand(int idx) {
        Node& node = nodes[static_cast<size_t>(idx)];
        bool has_goal_slot = false;
        for (const auto& s : node.biscuit) {
            for (const auto& g : goal) {
                if (s.opt_idx == g.opt_idx &&
                    get_tier(s.opt_idx, s.val) >= g.min_tier) {
                    has_goal_slot = true; break;
                }
            }
            if (has_goal_slot) break;
        }
        std::array<int, N_ACTIONS> acts{}; int n_acts = 0;
        for (int a = 0; a < N_ACTIONS; ++a) {
            if (!node.untried[static_cast<size_t>(a)] || !node.can_act(a)) continue;
            if (a >= 1 && !has_goal_slot) continue;
            if (a >= 1) {
                int pos = a - 1;
                bool is_goal_slot = false;
                for (const auto& g : goal) {
                    if (node.biscuit[static_cast<size_t>(pos)].opt_idx == g.opt_idx &&
                        get_tier(node.biscuit[static_cast<size_t>(pos)].opt_idx,
                            node.biscuit[static_cast<size_t>(pos)].val) >= g.min_tier) {
                        is_goal_slot = true; break;
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

    float rollout(int idx) {
        Biscuit b = nodes[static_cast<size_t>(idx)].biscuit;
        int bg = nodes[static_cast<size_t>(idx)].bg;
        int bl = nodes[static_cast<size_t>(idx)].bl;
        bool success = false;
        
        for (int step = 0; step < max_rollout; ++step) {
            if (check_goal(b, goal)) { success = true; break; }
            int a = -1;
            auto matched = get_matched_slots(b, goal);
            std::array<int, 4> unmatched_actions{};
            int unmatched_cnt = 0;
            for (int i = 0; i < 4; ++i)
                if (!matched[i]) unmatched_actions[unmatched_cnt++] = i + 1;
            float prog = static_cast<float>(4 - unmatched_cnt) / 4.0f;
            if (bl >= 1 && unmatched_cnt > 0 && (prog >= 0.5f || bg < 1))
                a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
            else if (bg >= 1 && bl >= 1) {
                if (tl_rng.next_int(10) < 8) a = 0;
                else if (unmatched_cnt > 0) a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
                else a = 0;
            }
            else if (bg >= 1) a = 0;
            else if (bl >= 1 && unmatched_cnt > 0)
                a = unmatched_actions[tl_rng.next_int(unmatched_cnt)];
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

// ── 고성능 락프리 링버퍼 (MPSC) ────────────────────────────
template <typename T, size_t Capacity>
class LockFreeMPSCQueue {
private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    Cell buffer[Capacity];
    
    alignas(64) std::atomic<size_t> head{0};
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
        Cell* cell;
        size_t h = head.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer[h & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(h + 1);
            if (diff == 0) {
                if (head.compare_exchange_weak(h, h + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; 
            } else {
                h = head.load(std::memory_order_relaxed);
            }
        }
        data = cell->data;
        cell->sequence.store(h + Capacity, std::memory_order_release);
        return true;
    }
};

// ── CSV 헤더 ──────────────────────────────────────────────
static const std::string CSV_HEADER =
"tier_0,tier_1,tier_2,tier_3,tier_4,progress,budget_g,budget_l,"
"opt0_tier,opt1_tier,opt2_tier,opt3_tier,opt4_tier,opt5_tier,"
"opt6_tier,opt7_tier,opt8_tier,opt9_tier,opt10_tier,opt11_tier,"
"best_action,best_q,q_general,q_light_avg,"
"goal_id,resource_tier,biscuit_id,matched_at_switch,bl_remaining_at_switch\n";

std::unordered_set<int> get_completed_biscuit_ids(const std::string& path) {
    std::unordered_set<int> completed;
    std::ifstream f(path);
    if (!f.good()) return completed;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() >= 3) {
            try {
                completed.insert(std::stoi(cols[cols.size() - 3]));
            } catch (...) {}
        }
    }
    return completed;
}

std::vector<Biscuit> generate_inits(const Biscuit& base, int n, std::mt19937& rng) {
    (void)rng;
    std::vector<Biscuit> inits(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (i % 3 == 0) {
            for (int p = 0; p < 4; ++p) {
                int opt = tl_rng.next_int(N_OPTS);
                int vidx = tl_rng.next_int(OPT_SIZES[static_cast<size_t>(opt)]);
                inits[static_cast<size_t>(i)][static_cast<size_t>(p)] =
                { opt, OPT_VALS[static_cast<size_t>(opt)][static_cast<size_t>(vidx)] };
            }
        }
        else {
            inits[static_cast<size_t>(i)] = base;
            int nr = tl_rng.next_int(3);
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

struct MCTSConfig {
    Biscuit init_biscuit = { Slot{ 0, 3.0f }, Slot{ 1, 5.0f }, Slot{ 3, 3.0f }, Slot{ 4, 3.0f } };
    Goal goal = { { 3, 2, 3 },{ 4, 2, 3 } };
    int budget_g = 90000;
    int budget_l = 3000;
    int n_starts = 100000;
    int n_sim = 500;
    int max_rollout = 300;
    int goal_id = 0;
    int resource_tier = 0;
    std::string output_path = "";
};

// ── 마감된 파이프라인 (완벽한 잔여 데이터 드레이닝 + 파일 예외 처리) ──
void run_mcts_pipeline(const MCTSConfig& cfg, const std::unordered_set<int>& completed_ids) {
    std::mt19937 rng{ 42u };
    auto inits = generate_inits(cfg.init_biscuit, cfg.n_starts, rng);

    std::atomic<int> done{ 0 };
    std::atomic<bool> producer_finished{ false };

    LockFreeMPSCQueue<Record, 65536> queue;

    auto t0 = std::chrono::steady_clock::now();

    // 백그라운드 I/O 스레드 (마감 완료: 드레이닝 및 예외 처리 포함)
    std::thread io_thread([&]() {
        std::ofstream f(cfg.output_path, std::ios::app);
        if (!f.is_open()) {
            std::cerr << "[Error] 출력 파일 오픈 실패: " << cfg.output_path << std::endl;
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
                  << "," << r.bl_remaining_at_switch << "\n";
            }
            f.flush(); // 디스크 쓰기 보장
        };

        Record rec;
        while (true) {
            if (queue.pop(rec)) {
                write_buffer.push_back(rec);
                if (write_buffer.size() >= 1000) {
                    flush_buffer(write_buffer);
                    write_buffer.clear();
                }
            } else {
                // 워커들이 모두 끝났다는 신호가 들어왔을 때
                if (producer_finished.load(std::memory_order_acquire)) {
                    // 큐에 남은 데이터가 완벽히 비워질 때까지 한 번 더 확인 (Double-check Draining)
                    if (!queue.pop(rec)) {
                        break; // 완전히 비었음이 확인되면 안전하게 탈출
                    }
                    write_buffer.push_back(rec);
                } else {
                    std::this_thread::yield();
                }
            }
        }

        // 마지막 잔여 버퍼 플러시
        flush_buffer(write_buffer);
    });

    const unsigned int num_threads = 2;
    std::vector<std::thread> workers;
    int chunk_size = cfg.n_starts / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_idx = static_cast<int>(t) * chunk_size;
        int end_idx = (t == num_threads - 1) ? cfg.n_starts : (static_cast<int>(t) + 1) * chunk_size;

        workers.emplace_back([&, start_idx, end_idx]() {
            std::vector<Node> local_node_pool;
            local_node_pool.reserve(2000000);

            for (int i = start_idx; i < end_idx; ++i) {
                if (completed_ids.find(i) != completed_ids.end()) {
                    ++done;
                    continue;
                }

                local_node_pool.clear();
                MCTSTree tree(local_node_pool, inits[i],
                    cfg.budget_g, cfg.budget_l, cfg.goal, cfg.max_rollout);
                tree.run(cfg.n_sim);

                auto recs = tree.extract_records(200);
                if (!recs.empty()) {
                    for (auto& r : recs) {
                        r.goal_id = cfg.goal_id;
                        r.resource_tier = cfg.resource_tier;
                        r.biscuit_id = i;
                        
                        while (!queue.push(r)) {
                            std::this_thread::yield();
                        }
                    }
                }

                int cnt = ++done;
                if (cnt % 10000 == 0)
                    std::cout << "  " << cnt << "/" << cfg.n_starts << std::endl;
            }
        });
    }

    // 워커 스레드 종료 대기 완료 후 프로듀서 마감 신호 전송
    for (auto& w : workers) if (w.joinable()) w.join();
    producer_finished.store(true, std::memory_order_release);

    // I/O 스레드가 남은 큐를 모두 비우고 종료될 때까지 안전하게 대기
    if (io_thread.joinable()) io_thread.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[MCTS] 탐색 완료: " << elapsed << "초\n";
}

// ── 메인 ──────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    build_opt_vals();

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

    for (int ti = 0; ti < (int)TIERS.size(); ++ti) {
        for (int gi = 0; gi < (int)GOALS.size(); ++gi) {
            std::string filename = "biscuit_tier_" + std::to_string(ti)
                + "_goal_" + std::to_string(gi) + ".csv";

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

            std::unordered_set<int> completed_ids = get_completed_biscuit_ids(filename);
            int total_completed = static_cast<int>(completed_ids.size());

            if (total_completed >= cfg.n_starts) {
                std::cout << "\n[Skip] 완료된 청크: " << filename << " (" << total_completed << "/" << cfg.n_starts << ")\n";
                continue;
            }

            if (total_completed == 0) {
                std::ofstream hdr(filename);
                hdr << CSV_HEADER;
                std::cout << "\n[New ] " << filename << " 생성\n";
            }
            else {
                std::cout << "\n[Resume] " << filename
                    << " (이미 완료된 항목: " << total_completed << "개, 빠진 구간 메우기 시작)\n";
            }

            std::cout << "[Tier " << ti << " Goal " << gi << " 시작]\n";
            run_mcts_pipeline(cfg, completed_ids);
        }
    }

    std::cout << "\n[All Done] 모든 청크 완료!\n";
    return 0;
}
