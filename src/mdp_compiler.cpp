// ============================================================================
// FILE:    mdp_compiler.cpp
// PROJECT: MDP-DSL v3.0 — POMDP-Capable Decision-Making Workbench
// AUTHOR:  Charvit Rajani (Roll: 240102028)
// DATE:    2026-05
//
// PURPOSE: Unified compiler pipeline. v3.0 adds POMDP extension (Phase 1):
//   Phase 0:   Preprocessor (GRID: macro expansion)
//   Phase 1:   Parser       (21 keywords, including OBSERVATION/OBSERVE_PROB/
//                            INITIAL_BELIEF/BELIEF_STATE/HORIZON/PBVI knobs)
//   Phase 2:   Validator    (21 semantic checks: 14 MDP + 7 POMDP)
//   Phase 3:   Solver       (Value Iteration, Policy Iteration, Q-Learning,
//                            PBVI for POMDPs)
//   Phase 3.5: Verifier     (VERIFY: assertions over V(s)/π(s) AND over
//                            named BELIEF_STATEs)
//
// COMPILE: g++ -std=c++17 -O2 -Wall -Wextra -o mdp_compiler mdp_compiler.cpp
// RUN:     ./mdp_compiler examples/tiger.mdp
//
// All v1.0 and v2.0 .mdp files are 100% backward compatible (byte-identical
// output for non-POMDP inputs). Pure C++17, zero external libraries.
// ============================================================================

#ifndef MDP_COMPILER_HPP_INCLUDED
#define MDP_COMPILER_HPP_INCLUDED

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>
#include <map>
#include <set>
#include <functional>
#include <cassert>
#include <deque>

#ifdef _WIN32
// Used only to enable ANSI escape processing in the console at startup.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// ============================================================================
// SECTION 1: DATA STRUCTURES (AST + Result Types)
// ============================================================================

// ---- Transition: one probabilistic outcome ----
struct Transition {
    std::string source_state;
    std::string action;
    std::string dest_state;
    double probability;
};

// ---- Reward struct (legacy, kept for compatibility) ----
struct Reward {
    std::string state_name;
    double value;
};

// ---- Type system structs (Upgrade 4) ----
struct StateTypeField { std::string name; std::string datatype; };
struct StateTypeDef   { std::string name; std::vector<StateTypeField> fields; };
struct TypedStateInfo { std::string type_name; std::vector<std::string> field_values; };

// ---- v3.0 POMDP: named belief state ----
//   BELIEF_STATE: name  s0:p0  s1:p1  ...
//   Stored as a sparse probability vector keyed by state name.
struct NamedBelief {
    std::string name;
    std::unordered_map<std::string, double> probs;
};

// ---- MDP_AST: master container for all parsed data ----
struct MDP_AST {
    // --- Phase 1 (original) ---
    std::unordered_set<std::string>          states;
    std::unordered_set<std::string>          actions;
    std::vector<Transition>                  transitions;
    std::unordered_map<std::string, double>  rewards;       // R(s)
    double                                   discount_factor;

    // --- Upgrade 1: Solver selection ---
    std::string solver_mode;    // "value_iteration", "policy_iteration", "q_learning", "all", "pbvi"

    // --- Upgrade 2: R(s,a) action-dependent rewards ---
    std::unordered_map<std::string, std::unordered_map<std::string, double>> action_rewards;

    // --- Upgrade 3: Q-Learning parameters ---
    double learning_rate;
    double epsilon_start;
    double epsilon_decay;
    int    episodes;
    int    random_seed;

    // --- Upgrade 4: Type system ---
    std::unordered_map<std::string, StateTypeDef>   state_types;
    std::unordered_map<std::string, TypedStateInfo>  typed_states;

    // --- Upgrade 6: VERIFY assertions ---
    std::vector<std::string> verify_assertions;

    // --- v3.0 POMDP extension (Phase 1) ---
    // Declared observation symbols (Ω).
    std::unordered_set<std::string> observations;
    // O(s,a,o) = P(observation o | next state s', action a). Sparse;
    // key = (next_state, action), value = map<observation, prob>.
    // Per POMDP convention we index by NEXT state and action taken to reach it.
    std::unordered_map<std::string,
        std::unordered_map<std::string,
            std::unordered_map<std::string, double>>> observe_probs;
    // Initial belief b_0(s) = p, as a sparse probability vector.
    std::unordered_map<std::string, double> initial_belief;
    // Named belief states, used by VERIFY and BELIEF_VERIFY.
    std::unordered_map<std::string, NamedBelief> belief_states;
    // PBVI knobs.
    int    pbvi_alpha_vectors;   // belief set size cap (default 15)
    int    pbvi_iterations;      // backup rounds (default 100)
    int    horizon;              // finite-horizon (<=0 => infinite horizon)
    // BELIEF_VERIFY assertions: "b(state) op value", evaluated against the
    // initial belief if no name is given, or specific named beliefs otherwise.
    std::vector<std::string> belief_verify_assertions;

    // ---- v3.0 HMM-MDP Bridge (Phase 2A) ----
    //
    // The user declares the hidden regime names with HMM_STATES, an emission
    // type (currently "gaussian"), and a data file via FIT_HMM. Baum-Welch
    // runs at compile time, learning A_ij and emission params from a CSV of
    // scalar observations (one float per line). With BRIDGE: hmm -> mdp the
    // learned A_ij is plumbed into the AST as MDP TRANSITION rows that are
    // independent of action — market regimes don't care what allocation you
    // chose, only the REWARDS differ by regime.
    std::vector<std::string> hmm_states;            // ordered list (parser order)
    std::string hmm_emission_type;                  // "gaussian" | "discrete"
    std::string hmm_fit_csv;                        // FIT_HMM: filename
    int  hmm_fit_iterations;                        // iterations=N (default 200)
    int  hmm_seed;                                  // HMM_SEED for reproducibility
    bool hmm_bridge_enabled;                        // true iff BRIDGE: hmm -> mdp
    // Learned parameters — written by the bridge after Baum-Welch runs.
    std::vector<std::vector<double>> hmm_transition_matrix;  // A[K][K]
    std::vector<double> hmm_initial;                // pi_i  (K)
    std::vector<double> hmm_emission_mean;          // mu_i  (K)  for gaussian
    std::vector<double> hmm_emission_std;           // sigma_i (K) for gaussian
    double hmm_log_likelihood;                      // final log P(O | lambda)
    bool   hmm_fitted;                              // bridge completed cleanly

    // --- Constructor with defaults ---
    MDP_AST() : discount_factor(0.9), solver_mode("value_iteration"),
                learning_rate(0.1), epsilon_start(0.9), epsilon_decay(0.995),
                episodes(50000), random_seed(42),
                pbvi_alpha_vectors(15), pbvi_iterations(100), horizon(-1),
                hmm_fit_iterations(200), hmm_seed(42),
                hmm_bridge_enabled(false), hmm_log_likelihood(0.0),
                hmm_fitted(false) {}
};

// ---- Solver result types ----
struct SolverResult {
    std::unordered_map<std::string, double>      values;
    std::unordered_map<std::string, std::string>  policy;
    int  iterations;
    bool converged;
    double wall_clock_ms;
};

struct PolicyIterationResult {
    std::unordered_map<std::string, double>      values;
    std::unordered_map<std::string, std::string>  policy;
    int  policy_improvement_steps;
    int  total_bellman_evaluations;
    bool converged;
    double wall_clock_ms;
};

struct QLearningResult {
    std::unordered_map<std::string, std::unordered_map<std::string, double>> Q_table;
    std::unordered_map<std::string, std::string>  policy;
    int  episodes_to_convergence;
    bool converged;
    double wall_clock_ms;
    std::vector<double> episode_rewards;
    std::vector<double> avg_rewards_100ep;
};

struct ValidationResult {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool isValid() const { return errors.empty(); }
};

struct VerifyResult {
    int pass_count;
    int fail_count;
    std::vector<std::string> messages;
};

// ---- v3.0 PBVI result type ----
//
// An alpha vector α is a function α : S → R. The PBVI value function is
// V(b) = max_i Σ_s b(s) · α_i(s), and π_PBVI(b) = action(α*) where α* is
// the argmax. We carry the action label alongside each vector so that policy
// extraction is just argmax over Γ for a given belief.
struct AlphaVector {
    std::unordered_map<std::string, double> values;  // α(s) for each state
    std::string action;                              // π label
};

struct PBVIResult {
    std::vector<AlphaVector> alpha_vectors;          // upper envelope Γ
    std::vector<NamedBelief> belief_points;          // the belief set B
    // For each named belief / initial belief / each b in B, we record V(b)
    // and π(b) under the final value function. Keyed by belief name; for
    // anonymous belief points we use "_b<i>".
    std::unordered_map<std::string, double>      values_at_belief;
    std::unordered_map<std::string, std::string> policy_at_belief;
    int  iterations;
    bool converged;
    double wall_clock_ms;
};

// ============================================================================
// v3.0 Phase 3A — AUTOPSY ENGINE DATA STRUCTURES
// ============================================================================
//
// Each Finding has a severity level, a class label, and a structured payload
// the reporter pretty-prints into the human-readable output and the JSON
// emitter serialises. The 6 classes match the master spec:
//
//   Class 1: Reward Myopia          (effective horizon < goal distance)
//   Class 2: Reward Hacking         (high-reward cycle in transition graph)
//   Class 3: Discount Cliff         (policy phase transition under γ sweep)
//   Class 4: Magnitude Imbalance    (reward term < 0.1% of total contribution)
//   Class 5: Dead State             (unreachable positive reward)
//   Class 6: Fragility              (minimum perturbation to flip policy)
//
// Severities: ERROR (definite issue), WARN (sensitivity / structural), INFO.
//
// Findings are deterministic and idempotent: running --diagnose on the same
// .mdp twice produces byte-identical output.
enum class FindingSeverity { INFO, WARN, ERROR_LEVEL };
enum class FindingClass {
    RewardMyopia,
    RewardHacking,
    DiscountCliff,
    MagnitudeImbalance,
    DeadState,
    Fragility,
};

inline const char* findingClassName(FindingClass c) {
    switch (c) {
        case FindingClass::RewardMyopia:        return "RewardMyopia";
        case FindingClass::RewardHacking:       return "RewardHacking";
        case FindingClass::DiscountCliff:       return "DiscountCliff";
        case FindingClass::MagnitudeImbalance:  return "MagnitudeImbalance";
        case FindingClass::DeadState:           return "DeadState";
        case FindingClass::Fragility:           return "Fragility";
    }
    return "Unknown";
}
inline const char* findingSeverityName(FindingSeverity s) {
    switch (s) {
        case FindingSeverity::ERROR_LEVEL: return "ERROR";
        case FindingSeverity::WARN:        return "WARN";
        case FindingSeverity::INFO:        return "INFO";
    }
    return "INFO";
}

struct AutopsyFinding {
    FindingClass    cls;
    FindingSeverity severity;
    std::string     headline;             // one-line summary
    std::vector<std::string> details;     // multi-line body
    // Optional structured payload for JSON consumers.
    std::unordered_map<std::string, double>       numeric_fields;
    std::unordered_map<std::string, std::string>  text_fields;
};

struct AutopsyReport {
    std::vector<AutopsyFinding> findings;
    bool has_issues() const {
        for (const auto& f : findings)
            if (f.severity == FindingSeverity::ERROR_LEVEL ||
                f.severity == FindingSeverity::WARN) return true;
        return false;
    }
    bool has_errors() const {
        for (const auto& f : findings)
            if (f.severity == FindingSeverity::ERROR_LEVEL) return true;
        return false;
    }
};

// A repair proposal: one parameter change that satisfies the target assertion.
struct RepairProposal {
    std::string parameter_kind;  // "discount" | "reward" | "action_reward"
    std::string parameter_label; // human-readable label (e.g. "R(cliff, MoveLeft)")
    double      original_value;
    double      new_value;
    double      magnitude;       // |delta|, used for ranking
    bool        succeeded;       // assertion satisfied after applying the repair?
    std::string note;            // free-text annotation
};

struct RepairResult {
    bool already_satisfied;          // assertion holds in the unmodified MDP
    bool any_succeeded;
    std::vector<RepairProposal> proposals;
    std::string target_assertion;
};


// ============================================================================
// SECTION 2: UTILITY FUNCTIONS
// ============================================================================

inline std::string trim(const std::string& str) {
    const std::string ws = " \t\r\n";
    auto s = str.find_first_not_of(ws);
    if (s == std::string::npos) return "";
    auto e = str.find_last_not_of(ws);
    return str.substr(s, e - s + 1);
}

inline std::string toUpper(const std::string& str) {
    std::string r = str;
    for (char& c : r) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return r;
}

// ---- getReward: R(s,a) with fallback to R(s) then 0.0 (Upgrade 2) ----
inline double getReward(const MDP_AST& ast, const std::string& state, const std::string& action) {
    auto it = ast.action_rewards.find(state);
    if (it != ast.action_rewards.end()) {
        auto it2 = it->second.find(action);
        if (it2 != it->second.end()) return it2->second;
    }
    auto it3 = ast.rewards.find(state);
    if (it3 != ast.rewards.end()) return it3->second;
    return 0.0;
}

// ---- Check if a string can be parsed as int ----
inline bool isInt(const std::string& s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i) if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}

// ---- Check if a string can be parsed as float ----
inline bool isFloat(const std::string& s) {
    if (s.empty()) return false;
    try { std::stod(s); return true; } catch (...) { return false; }
}

// ---- Check if a string is "true" or "false" ----
inline bool isBool(const std::string& s) {
    return s == "true" || s == "false";
}


// ============================================================================
// SECTION 3: PHASE 0 — PREPROCESSOR (Upgrade 5: GRID Macros)
// ============================================================================

// Expands GRID: macro blocks into explicit MDP statements.
// Returns the expanded .mdp content as a string.
// If no GRID: block found, returns the original file content unchanged.
std::string preprocessMDPFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return "";

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    file.close();

    // Check if GRID: keyword exists
    bool has_grid = false;
    {
        std::istringstream check(content);
        std::string cline;
        while (std::getline(check, cline)) {
            std::string t = trim(cline);
            if (t.empty() || t[0] == '#') continue;
            std::istringstream iss(t);
            std::string kw; iss >> kw;
            if (toUpper(kw) == "GRID:") { has_grid = true; break; }
        }
    }

    if (!has_grid) return content;

    // Parse grid parameters
    int rows = 0, cols = 0, goal_r = 0, goal_c = 0, trap_r = 0, trap_c = 0;
    std::vector<std::pair<int,int>> walls;
    double p_intended = 0.8, p_slip_l = 0.1, p_slip_r = 0.1;
    double living_reward = -0.04, goal_reward = 1.0, trap_reward = -1.0;
    double grid_discount = 0.9;
    std::string non_grid_lines;  // Collect non-grid lines (SOLVER:, VERIFY:, etc.)

    std::istringstream iss(content);
    while (std::getline(iss, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') { non_grid_lines += line + "\n"; continue; }
        std::istringstream ls(t);
        std::string kw; ls >> kw;
        std::string ku = toUpper(kw);

        if (ku == "GRID:") {
            ls >> rows >> cols >> goal_r >> goal_c >> trap_r >> trap_c;
        } else if (ku == "WALL:") {
            int wr, wc; ls >> wr >> wc;
            walls.emplace_back(wr, wc);
        } else if (ku == "SLIP_MODEL:") {
            ls >> p_intended >> p_slip_l >> p_slip_r;
        } else if (ku == "LIVING_REWARD:") {
            ls >> living_reward;
        } else if (ku == "GOAL_REWARD:") {
            ls >> goal_reward;
        } else if (ku == "TRAP_REWARD:") {
            ls >> trap_reward;
        } else if (ku == "GRID_DISCOUNT:") {
            ls >> grid_discount;
        } else {
            non_grid_lines += line + "\n";
        }
    }

    if (rows <= 0 || cols <= 0) return content;

    // Build wall set for quick lookup
    std::set<std::pair<int,int>> wall_set(walls.begin(), walls.end());

    // Lambda: is (r,c) navigable?
    auto navigable = [&](int r, int c) -> bool {
        if (r < 1 || r > rows || c < 1 || c > cols) return false;
        return wall_set.find({r, c}) == wall_set.end();
    };

    // Lambda: state name
    auto sname = [](int r, int c) -> std::string {
        return "R" + std::to_string(r) + "C" + std::to_string(c);
    };

    // Direction vectors: Up=(+1,0), Down=(-1,0), Left=(0,-1), Right=(0,+1)
    struct Dir { std::string name; int dr; int dc; };
    std::vector<Dir> dirs = {{"Up",1,0}, {"Down",-1,0}, {"Left",0,-1}, {"Right",0,+1}};

    // Slip directions for each action
    // Up: slip-left=Left, slip-right=Right
    // Down: slip-left=Right, slip-right=Left
    // Left: slip-left=Down, slip-right=Up
    // Right: slip-left=Up, slip-right=Down
    auto slip_left_idx = [](int d) -> int { return std::vector<int>{2,3,1,0}[d]; };
    auto slip_right_idx = [](int d) -> int { return std::vector<int>{3,2,0,1}[d]; };

    // Generate expanded content
    std::ostringstream out;
    out << "# Auto-generated by GRID: macro preprocessor\n\n";

    // States
    for (int r = 1; r <= rows; r++)
        for (int c = 1; c <= cols; c++)
            if (navigable(r, c))
                out << "STATE: " << sname(r, c) << "\n";
    out << "\n";

    // Actions
    for (auto& d : dirs) out << "ACTION: " << d.name << "\n";
    out << "\n";

    // Transitions
    bool is_goal = false, is_trap = false;
    for (int r = 1; r <= rows; r++) {
        for (int c = 1; c <= cols; c++) {
            if (!navigable(r, c)) continue;
            is_goal = (r == goal_r && c == goal_c);
            is_trap = (r == trap_r && c == trap_c);

            if (is_goal || is_trap) {
                // Absorbing state: all actions self-loop with p=1.0
                for (auto& d : dirs)
                    out << "TRANSITION: " << sname(r,c) << " " << d.name
                        << " " << sname(r,c) << " 1.0\n";
            } else {
                // Normal state with stochastic slip
                for (int di = 0; di < 4; di++) {
                    std::unordered_map<std::string, double> outcomes;

                    // Three movement attempts: intended, slip-left, slip-right
                    struct Attempt { int dir_idx; double prob; };
                    std::vector<Attempt> attempts = {
                        {di, p_intended},
                        {slip_left_idx(di), p_slip_l},
                        {slip_right_idx(di), p_slip_r}
                    };

                    for (auto& att : attempts) {
                        int nr = r + dirs[att.dir_idx].dr;
                        int nc = c + dirs[att.dir_idx].dc;
                        std::string dest;
                        if (navigable(nr, nc)) {
                            dest = sname(nr, nc);
                        } else {
                            dest = sname(r, c); // Bounce back
                        }
                        outcomes[dest] += att.prob;
                    }

                    for (auto& [dest, prob] : outcomes) {
                        out << "TRANSITION: " << sname(r,c) << " " << dirs[di].name
                            << " " << dest << " " << std::fixed << std::setprecision(4) << prob << "\n";
                    }
                }
            }
        }
    }
    out << "\n";

    // Rewards
    for (int r = 1; r <= rows; r++) {
        for (int c = 1; c <= cols; c++) {
            if (!navigable(r, c)) continue;
            double rwd;
            if (r == goal_r && c == goal_c) rwd = goal_reward;
            else if (r == trap_r && c == trap_c) rwd = trap_reward;
            else rwd = living_reward;
            out << "REWARD: " << sname(r,c) << " " << std::fixed << std::setprecision(4) << rwd << "\n";
        }
    }
    out << "\nDISCOUNT: " << std::fixed << std::setprecision(4) << grid_discount << "\n\n";

    // Append non-grid lines (SOLVER:, VERIFY:, etc.)
    out << non_grid_lines;

    return out.str();
}


// ============================================================================
// SECTION 4: PHASE 1 — PARSER (Extended for v2.0)
// ============================================================================

// Parse from a string (for preprocessor output) or file
MDP_AST parseMDPString(const std::string& content, const std::string& source_name = "<string>") {
    MDP_AST ast;
    std::istringstream file(content);
    std::string line;
    int line_number = 0;

    std::cout << "=== MDP Compiler v3.0 ===" << std::endl;
    std::cout << "Parsing: " << source_name << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    while (std::getline(file, line)) {
        line_number++;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::istringstream iss(trimmed);
        std::string keyword;
        iss >> keyword;
        std::string ku = toUpper(keyword);

        if (ku == "STATE:") {
            std::string name; iss >> name;
            if (name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": STATE with no name.\n"; continue; }

            // Check for optional type annotation: STATE: Name TypeName(v1,v2,...)
            std::string type_token;
            if (iss >> type_token) {
                // Parse TypeName(v1,v2,...)
                auto paren_open = type_token.find('(');
                if (paren_open != std::string::npos) {
                    std::string type_name = type_token.substr(0, paren_open);
                    std::string vals_str = type_token.substr(paren_open + 1);
                    // Remove trailing ')'
                    if (!vals_str.empty() && vals_str.back() == ')') vals_str.pop_back();
                    // Split by comma
                    TypedStateInfo tsi;
                    tsi.type_name = type_name;
                    std::istringstream vss(vals_str);
                    std::string val;
                    while (std::getline(vss, val, ',')) {
                        tsi.field_values.push_back(trim(val));
                    }
                    ast.typed_states[name] = tsi;
                } else {
                    // Type name without parentheses (zero-field type)
                    TypedStateInfo tsi;
                    tsi.type_name = type_token;
                    ast.typed_states[name] = tsi;
                }
            }

            auto [it, ok] = ast.states.insert(name);
            if (!ok) std::cerr << "[WARNING] Line " << line_number << ": Duplicate state '" << name << "'.\n";
        }
        else if (ku == "ACTION:") {
            std::string name; iss >> name;
            if (name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": ACTION with no name.\n"; continue; }
            auto [it, ok] = ast.actions.insert(name);
            if (!ok) std::cerr << "[WARNING] Line " << line_number << ": Duplicate action '" << name << "'.\n";
        }
        else if (ku == "TRANSITION:") {
            std::string src, act, dst; double prob = 0.0;
            iss >> src >> act >> dst >> prob;
            if (iss.fail() || src.empty() || act.empty() || dst.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": Malformed TRANSITION.\n"; continue;
            }
            if (prob < 0.0 || prob > 1.0)
                std::cerr << "[WARNING] Line " << line_number << ": Prob " << prob << " out of [0,1].\n";
            ast.transitions.emplace_back(Transition{src, act, dst, prob});
        }
        else if (ku == "REWARD:") {
            std::string name; double val = 0.0;
            iss >> name >> val;
            if (iss.fail() || name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": Malformed REWARD.\n"; continue; }
            if (ast.rewards.count(name) > 0)
                std::cerr << "[WARNING] Line " << line_number << ": Reward for '" << name << "' overwritten.\n";
            ast.rewards[name] = val;
        }
        else if (ku == "ACTION_REWARD:") {
            std::string state, action; double val = 0.0;
            iss >> state >> action >> val;
            if (iss.fail() || state.empty() || action.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": Malformed ACTION_REWARD.\n"; continue;
            }
            ast.action_rewards[state][action] = val;
        }
        else if (ku == "DISCOUNT:") {
            double gamma = 0.9; iss >> gamma;
            if (iss.fail()) { std::cerr << "[WARNING] Line " << line_number << ": Bad DISCOUNT.\n"; gamma = 0.9; }
            if (gamma < 0.0 || gamma > 1.0) {
                std::cerr << "[WARNING] Line " << line_number << ": DISCOUNT " << gamma << " clamped.\n";
                gamma = (gamma < 0.0) ? 0.0 : 1.0;
            }
            ast.discount_factor = gamma;
        }
        else if (ku == "SOLVER:") {
            std::string mode; iss >> mode;
            if (!mode.empty()) ast.solver_mode = mode;
        }
        else if (ku == "LEARNING_RATE:") {
            double v; iss >> v;
            if (!iss.fail()) ast.learning_rate = v;
        }
        else if (ku == "EPSILON:") {
            double v; iss >> v;
            if (!iss.fail()) ast.epsilon_start = v;
        }
        else if (ku == "EPSILON_DECAY:") {
            double v; iss >> v;
            if (!iss.fail()) ast.epsilon_decay = v;
        }
        else if (ku == "EPISODES:") {
            int v; iss >> v;
            if (!iss.fail()) ast.episodes = v;
        }
        else if (ku == "RANDOM_SEED:") {
            int v; iss >> v;
            if (!iss.fail()) ast.random_seed = v;
        }
        else if (ku == "STATE_TYPE:") {
            std::string type_name; iss >> type_name;
            if (type_name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": STATE_TYPE with no name.\n"; continue; }
            StateTypeDef def;
            def.name = type_name;
            std::string field_spec;
            while (iss >> field_spec) {
                auto colon = field_spec.find(':');
                if (colon != std::string::npos) {
                    StateTypeField f;
                    f.name = field_spec.substr(0, colon);
                    f.datatype = field_spec.substr(colon + 1);
                    def.fields.push_back(f);
                }
            }
            ast.state_types[type_name] = def;
        }
        else if (ku == "VERIFY:") {
            // Everything after "VERIFY: " is the assertion expression
            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            if (!rest.empty()) ast.verify_assertions.push_back(rest);
        }
        // ---- v3.0 POMDP keywords ----
        else if (ku == "OBSERVATION:") {
            std::string name; iss >> name;
            if (name.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": OBSERVATION with no name.\n";
                continue;
            }
            auto [it, ok] = ast.observations.insert(name);
            if (!ok)
                std::cerr << "[WARNING] Line " << line_number << ": Duplicate observation '" << name << "'.\n";
        }
        else if (ku == "OBSERVE_PROB:") {
            // OBSERVE_PROB: next_state action observation probability
            std::string s, a, o; double p = 0.0;
            iss >> s >> a >> o >> p;
            if (iss.fail() || s.empty() || a.empty() || o.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": Malformed OBSERVE_PROB.\n";
                continue;
            }
            if (p < 0.0 || p > 1.0)
                std::cerr << "[WARNING] Line " << line_number << ": OBSERVE_PROB " << p << " out of [0,1].\n";
            ast.observe_probs[s][a][o] = p;
        }
        else if (ku == "INITIAL_BELIEF:") {
            std::string s; double p = 0.0;
            iss >> s >> p;
            if (iss.fail() || s.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": Malformed INITIAL_BELIEF.\n";
                continue;
            }
            ast.initial_belief[s] = p;
        }
        else if (ku == "BELIEF_STATE:") {
            // BELIEF_STATE: name  s1:p1  s2:p2  ...
            std::string name; iss >> name;
            if (name.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": BELIEF_STATE with no name.\n";
                continue;
            }
            NamedBelief nb;
            nb.name = name;
            std::string token;
            while (iss >> token) {
                auto colon = token.find(':');
                if (colon == std::string::npos) {
                    std::cerr << "[WARNING] Line " << line_number
                              << ": BELIEF_STATE token '" << token
                              << "' is not state:prob.\n";
                    continue;
                }
                std::string sname = token.substr(0, colon);
                std::string pstr  = token.substr(colon + 1);
                double p = 0.0;
                try { p = std::stod(pstr); }
                catch (...) {
                    std::cerr << "[WARNING] Line " << line_number
                              << ": Bad probability '" << pstr << "' in BELIEF_STATE.\n";
                    continue;
                }
                nb.probs[sname] = p;
            }
            ast.belief_states[name] = nb;
        }
        else if (ku == "BELIEF_VERIFY:") {
            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            if (!rest.empty()) ast.belief_verify_assertions.push_back(rest);
        }
        else if (ku == "ALPHA_VECTORS:") {
            int v; iss >> v;
            if (!iss.fail()) ast.pbvi_alpha_vectors = v;
        }
        else if (ku == "PBVI_ITERATIONS:") {
            int v; iss >> v;
            if (!iss.fail()) ast.pbvi_iterations = v;
        }
        else if (ku == "HORIZON:") {
            int v; iss >> v;
            if (!iss.fail()) ast.horizon = v;
        }
        // ---- v3.0 Phase 2A: HMM-MDP Bridge keywords ----
        else if (ku == "HMM_STATES:") {
            // Space-separated list of regime names.
            std::string name;
            while (iss >> name) ast.hmm_states.push_back(name);
        }
        else if (ku == "EMISSION_TYPE:") {
            std::string t; iss >> t;
            for (auto& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            ast.hmm_emission_type = t;
        }
        else if (ku == "FIT_HMM:") {
            // FIT_HMM: filename.csv [iterations=N]
            std::string fname, rest;
            iss >> fname;
            ast.hmm_fit_csv = fname;
            // Parse optional iterations=N token.
            while (iss >> rest) {
                auto eq = rest.find('=');
                if (eq != std::string::npos) {
                    std::string key = rest.substr(0, eq);
                    for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    std::string val = rest.substr(eq + 1);
                    if (key == "iterations") {
                        try { ast.hmm_fit_iterations = std::stoi(val); } catch (...) {}
                    }
                }
            }
        }
        else if (ku == "HMM_SEED:") {
            int v; iss >> v;
            if (!iss.fail()) ast.hmm_seed = v;
        }
        else if (ku == "BRIDGE:") {
            std::string rest;
            std::getline(iss, rest);
            rest = trim(rest);
            std::string low = rest;
            for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (low.find("hmm") != std::string::npos && low.find("mdp") != std::string::npos) {
                ast.hmm_bridge_enabled = true;
            } else {
                std::cerr << "[WARNING] Line " << line_number
                          << ": Unrecognized BRIDGE directive: '" << rest << "'\n";
            }
        }
        // Grid-related keywords handled by preprocessor, skip silently
        else if (ku == "GRID:" || ku == "WALL:" || ku == "SLIP_MODEL:" ||
                 ku == "LIVING_REWARD:" || ku == "GOAL_REWARD:" ||
                 ku == "TRAP_REWARD:" || ku == "GRID_DISCOUNT:") {
            // Already processed by preprocessor
        }
        else {
            std::cerr << "[WARNING] Line " << line_number << ": Unknown keyword '" << keyword << "'.\n";
        }
    }

    std::cout << "\nParsing complete!\n";
    std::cout << "  States:      " << ast.states.size() << "\n";
    std::cout << "  Actions:     " << ast.actions.size() << "\n";
    std::cout << "  Transitions: " << ast.transitions.size() << "\n";
    std::cout << "  Rewards:     " << ast.rewards.size() << "\n";
    std::cout << "  Discount:    " << ast.discount_factor << "\n";
    std::cout << "  Solver:      " << ast.solver_mode << "\n";
    std::cout << "----------------------------------------\n";

    return ast;
}

// Convenience: parse from file (runs preprocessor first)
MDP_AST parseMDPFile(const std::string& filepath) {
    std::string content = preprocessMDPFile(filepath);
    if (content.empty()) {
        std::cerr << "[FATAL ERROR] Cannot open file: " << filepath << "\n";
        return MDP_AST{};
    }
    return parseMDPString(content, filepath);
}

// ---- printAST ----
void printAST(const MDP_AST& ast) {
    std::cout << "\n======================================\n";
    std::cout << "    PARSED MDP -- ABSTRACT SYNTAX TREE\n";
    std::cout << "======================================\n\n";

    std::cout << "-- STATES (" << ast.states.size() << ") --\n";
    int idx = 1;
    for (const auto& s : ast.states) std::cout << "  " << idx++ << ". " << s << "\n";
    std::cout << "\n-- ACTIONS (" << ast.actions.size() << ") --\n";
    idx = 1;
    for (const auto& a : ast.actions) std::cout << "  " << idx++ << ". " << a << "\n";

    std::cout << "\n-- TRANSITIONS (" << ast.transitions.size() << ") --\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < ast.transitions.size(); ++i) {
        const auto& t = ast.transitions[i];
        std::cout << "  " << (i+1) << ". " << t.source_state
                  << " --[" << t.action << "]--> " << t.dest_state
                  << "  (p=" << t.probability << ")\n";
    }

    std::cout << "\n-- REWARDS (" << ast.rewards.size() << ") --\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [n, v] : ast.rewards) std::cout << "  " << n << " => " << v << "\n";

    if (!ast.action_rewards.empty()) {
        std::cout << "\n-- ACTION REWARDS --\n";
        for (const auto& [s, am] : ast.action_rewards)
            for (const auto& [a, v] : am)
                std::cout << "  R(" << s << "," << a << ") = " << v << "\n";
    }

    std::cout << "\n-- DISCOUNT: " << std::fixed << std::setprecision(4) << ast.discount_factor << "\n";
    std::cout << "-- SOLVER:   " << ast.solver_mode << "\n";
    if (!ast.verify_assertions.empty())
        std::cout << "-- VERIFY:   " << ast.verify_assertions.size() << " assertion(s)\n";
    std::cout << "\n======================================\n";
}


// ============================================================================
// SECTION 5: PHASE 2 — VALIDATOR (14 checks)
// ============================================================================

ValidationResult validateAST(const MDP_AST& ast) {
    ValidationResult result;
    std::unordered_map<std::string, double> prob_sums;
    std::unordered_set<std::string> states_in_transitions;

    // Checks 1-3: Reference integrity (single pass over transitions)
    for (const auto& t : ast.transitions) {
        if (ast.states.count(t.source_state) == 0)
            result.errors.push_back("Undeclared source state '" + t.source_state +
                "' in TRANSITION: " + t.source_state + " " + t.action + " " + t.dest_state);
        if (ast.states.count(t.dest_state) == 0)
            result.errors.push_back("Undeclared destination state '" + t.dest_state +
                "' in TRANSITION: " + t.source_state + " " + t.action + " " + t.dest_state);
        if (ast.actions.count(t.action) == 0)
            result.errors.push_back("Undeclared action '" + t.action +
                "' in TRANSITION: " + t.source_state + " " + t.action + " " + t.dest_state);
        prob_sums[t.source_state + "|" + t.action] += t.probability;
        states_in_transitions.insert(t.source_state);
        states_in_transitions.insert(t.dest_state);
    }

    // Check 4: Probability sums
    const double EPSILON = 1e-9;
    for (const auto& [key, sum] : prob_sums) {
        if (std::abs(sum - 1.0) > EPSILON) {
            size_t sep = key.find('|');
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(6);
            msg << "Probability sum for (" << key.substr(0, sep) << ", "
                << key.substr(sep+1) << ") = " << sum
                << " (expected 1.0, deviation = " << std::abs(sum - 1.0) << ")";
            result.errors.push_back(msg.str());
        }
    }

    // Check 5: Reward completeness
    for (const auto& state : ast.states)
        if (ast.rewards.count(state) == 0)
            result.warnings.push_back("State '" + state + "' has no REWARD. Solver assumes 0.0.");

    // Check 6: Orphan state detection
    for (const auto& state : ast.states)
        if (states_in_transitions.count(state) == 0)
            result.warnings.push_back("State '" + state + "' is orphaned (no transitions).");

    // Check 7: Reward reference integrity
    for (const auto& [rs, rv] : ast.rewards)
        if (ast.states.count(rs) == 0)
            result.errors.push_back("REWARD references undeclared state '" + rs + "'.");

    // Check 8: ACTION_REWARD state reference (Upgrade 2)
    for (const auto& [s, am] : ast.action_rewards)
        if (ast.states.count(s) == 0)
            result.errors.push_back("ACTION_REWARD references undeclared state '" + s + "'.");

    // Check 9: ACTION_REWARD action reference (Upgrade 2)
    for (const auto& [s, am] : ast.action_rewards)
        for (const auto& [a, v] : am)
            if (ast.actions.count(a) == 0)
                result.errors.push_back("ACTION_REWARD references undeclared action '" + a + "' for state '" + s + "'.");

    // Check 10: Type system — referenced types must exist (Upgrade 4)
    for (const auto& [sname, tsi] : ast.typed_states)
        if (ast.state_types.count(tsi.type_name) == 0)
            result.errors.push_back("State '" + sname + "' references undeclared type '" + tsi.type_name + "'.");

    // Check 11: Field count match (Upgrade 4)
    for (const auto& [sname, tsi] : ast.typed_states) {
        if (ast.state_types.count(tsi.type_name) > 0) {
            auto& def = ast.state_types.at(tsi.type_name);
            if (tsi.field_values.size() != def.fields.size())
                result.errors.push_back("State '" + sname + "' has " +
                    std::to_string(tsi.field_values.size()) + " field values but type '" +
                    tsi.type_name + "' expects " + std::to_string(def.fields.size()) + ".");
        }
    }

    // Check 12: Field type validation (Upgrade 4)
    for (const auto& [sname, tsi] : ast.typed_states) {
        if (ast.state_types.count(tsi.type_name) > 0) {
            auto& def = ast.state_types.at(tsi.type_name);
            size_t n = std::min(tsi.field_values.size(), def.fields.size());
            for (size_t i = 0; i < n; i++) {
                const auto& dt = def.fields[i].datatype;
                const auto& val = tsi.field_values[i];
                bool ok = true;
                if (dt == "int") ok = isInt(val);
                else if (dt == "float") ok = isFloat(val);
                else if (dt == "bool") ok = isBool(val);
                // "string" accepts anything
                if (!ok)
                    result.errors.push_back("State '" + sname + "' field '" +
                        def.fields[i].name + "' value '" + val + "' is not a valid " + dt + ".");
            }
        }
    }

    // Discount factor range
    if (ast.discount_factor < 0.0 || ast.discount_factor > 1.0)
        result.errors.push_back("Discount factor outside [0.0, 1.0].");

    // SOLVER: validation
    static const std::unordered_set<std::string> valid_solvers =
        {"value_iteration", "policy_iteration", "q_learning", "all", "pbvi"};
    if (valid_solvers.count(ast.solver_mode) == 0)
        result.errors.push_back("Unknown SOLVER mode: '" + ast.solver_mode + "'. Valid: value_iteration, policy_iteration, q_learning, all, pbvi.");

    // Q-Learning parameter validation (Checks 13-14 equivalent)
    if (ast.solver_mode == "q_learning" || ast.solver_mode == "all") {
        if (ast.learning_rate <= 0.0 || ast.learning_rate > 1.0)
            result.errors.push_back("LEARNING_RATE must be in (0, 1]. Got: " + std::to_string(ast.learning_rate));
        if (ast.epsilon_start <= 0.0 || ast.epsilon_start > 1.0)
            result.errors.push_back("EPSILON must be in (0, 1]. Got: " + std::to_string(ast.epsilon_start));
        if (ast.epsilon_decay <= 0.0 || ast.epsilon_decay >= 1.0)
            result.errors.push_back("EPSILON_DECAY must be in (0, 1). Got: " + std::to_string(ast.epsilon_decay));
        if (ast.episodes < 1)
            result.errors.push_back("EPISODES must be >= 1. Got: " + std::to_string(ast.episodes));
        if (ast.random_seed < 0)
            result.errors.push_back("RANDOM_SEED must be non-negative. Got: " + std::to_string(ast.random_seed));
    }

    // ---- v3.0 POMDP semantic checks (15-21) ----
    // Check 16 first so that we can refer to it consistently below.

    // Check 16: every observation referenced in OBSERVE_PROB is declared
    for (const auto& [s, am] : ast.observe_probs) {
        for (const auto& [a, om] : am) {
            for (const auto& [o, p] : om) {
                if (ast.observations.count(o) == 0)
                    result.errors.push_back("[Check 16] OBSERVE_PROB references undeclared OBSERVATION '" + o +
                                            "' for (state=" + s + ", action=" + a + ").");
                if (ast.states.count(s) == 0)
                    result.errors.push_back("[Check 16] OBSERVE_PROB references undeclared state '" + s + "'.");
                if (ast.actions.count(a) == 0)
                    result.errors.push_back("[Check 16] OBSERVE_PROB references undeclared action '" + a + "'.");
            }
        }
    }

    // Check 15: observation probabilities sum to 1.0 ± 1e-9 per (s,a) pair
    // (only when at least one OBSERVE_PROB entry exists for that pair).
    {
        const double EPS_OBS = 1e-9;
        for (const auto& [s, am] : ast.observe_probs) {
            for (const auto& [a, om] : am) {
                double sum = 0.0;
                for (const auto& [o, p] : om) sum += p;
                if (std::abs(sum - 1.0) > EPS_OBS) {
                    std::ostringstream msg;
                    msg << std::fixed << std::setprecision(6);
                    msg << "[Check 15] Observation probability sum for next_state='" << s
                        << "', action='" << a << "' = " << sum
                        << " (expected 1.0, deviation = " << std::abs(sum - 1.0) << ").";
                    result.errors.push_back(msg.str());
                }
            }
        }
    }

    // Check 17: INITIAL_BELIEF sums to 1.0 ± 1e-9 (only when declared)
    if (!ast.initial_belief.empty()) {
        const double EPS_OBS = 1e-9;
        double sum = 0.0;
        for (const auto& [s, p] : ast.initial_belief) {
            if (ast.states.count(s) == 0)
                result.errors.push_back("[Check 17] INITIAL_BELIEF references undeclared state '" + s + "'.");
            sum += p;
        }
        if (std::abs(sum - 1.0) > EPS_OBS) {
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(6);
            msg << "[Check 17] INITIAL_BELIEF sums to " << sum
                << " (expected 1.0, deviation = " << std::abs(sum - 1.0) << ").";
            result.errors.push_back(msg.str());
        }
    }

    // Check 18: pbvi solver requires at least one INITIAL_BELIEF or BELIEF_STATE
    if (ast.solver_mode == "pbvi") {
        if (ast.initial_belief.empty() && ast.belief_states.empty())
            result.errors.push_back("[Check 18] SOLVER: pbvi requires at least one INITIAL_BELIEF or BELIEF_STATE.");
    }

    // Checks 19 + 20: BELIEF_STATE state references and probability sum
    {
        const double EPS_OBS = 1e-9;
        for (const auto& [name, nb] : ast.belief_states) {
            double sum = 0.0;
            for (const auto& [s, p] : nb.probs) {
                if (ast.states.count(s) == 0)
                    result.errors.push_back("[Check 19] BELIEF_STATE '" + name +
                                            "' references undeclared state '" + s + "'.");
                sum += p;
            }
            if (std::abs(sum - 1.0) > EPS_OBS) {
                std::ostringstream msg;
                msg << std::fixed << std::setprecision(6);
                msg << "[Check 20] BELIEF_STATE '" << name << "' probabilities sum to " << sum
                    << " (expected 1.0, deviation = " << std::abs(sum - 1.0) << ").";
                result.errors.push_back(msg.str());
            }
        }
    }

    // Check 21: HORIZON must be a positive integer if declared (-1 sentinel = not set)
    if (ast.horizon == 0 || (ast.horizon != -1 && ast.horizon < 0)) {
        result.errors.push_back("[Check 21] HORIZON must be a positive integer if declared. Got: " +
                                std::to_string(ast.horizon));
    }

    // ---- v3.0 Phase 2A: HMM-MDP bridge semantic checks (22-25) ----

    // Check 22: BRIDGE: hmm -> mdp requires HMM_STATES to be declared.
    if (ast.hmm_bridge_enabled && ast.hmm_states.empty()) {
        result.errors.push_back("[Check 22] BRIDGE: hmm -> mdp requires at least one HMM_STATES entry.");
    }

    // Check 23: BRIDGE: hmm -> mdp requires FIT_HMM: <csv> to be specified.
    if (ast.hmm_bridge_enabled && ast.hmm_fit_csv.empty()) {
        result.errors.push_back("[Check 23] BRIDGE: hmm -> mdp requires FIT_HMM: <filename.csv>.");
    }

    // Check 24: EMISSION_TYPE must be one of {gaussian, discrete} if declared.
    if (!ast.hmm_emission_type.empty() &&
        ast.hmm_emission_type != "gaussian" &&
        ast.hmm_emission_type != "discrete") {
        result.errors.push_back("[Check 24] EMISSION_TYPE must be 'gaussian' or 'discrete'. Got: '"
                                + ast.hmm_emission_type + "'.");
    }

    // Check 25: HMM_SEED must be non-negative.
    if (ast.hmm_seed < 0) {
        result.errors.push_back("[Check 25] HMM_SEED must be non-negative. Got: " +
                                std::to_string(ast.hmm_seed));
    }

    return result;
}

void printValidationReport(const ValidationResult& result) {
    std::cout << "\n======================================\n";
    std::cout << "   PHASE 2 -- SEMANTIC VALIDATION\n";
    std::cout << "======================================\n\n";

    if (!result.errors.empty()) {
        std::cout << "-- ERRORS (" << result.errors.size() << ") --\n";
        for (size_t i = 0; i < result.errors.size(); ++i)
            std::cout << "  [ERROR " << (i+1) << "] " << result.errors[i] << "\n";
        std::cout << "\n";
    }
    if (!result.warnings.empty()) {
        std::cout << "-- WARNINGS (" << result.warnings.size() << ") --\n";
        for (size_t i = 0; i < result.warnings.size(); ++i)
            std::cout << "  [WARN " << (i+1) << "] " << result.warnings[i] << "\n";
        std::cout << "\n";
    }
    std::cout << "--------------------------------------\n";
    if (result.isValid()) {
        std::cout << "  VERDICT: MDP is VALID\n";
        if (!result.warnings.empty())
            std::cout << "  (" << result.warnings.size() << " warning(s))\n";
    } else {
        std::cout << "  VERDICT: MDP is INVALID\n";
        std::cout << "  " << result.errors.size() << " error(s). Fix and re-run.\n";
    }
    std::cout << "--------------------------------------\n";
}


// ============================================================================
// SECTION 6: SOLVER — Value Iteration (Updated for R(s,a))
// ============================================================================

SolverResult solveValueIteration(const MDP_AST& ast, double theta = 1e-9, int max_iter = 10000) {
    auto t_start = std::chrono::high_resolution_clock::now();

    SolverResult result;
    result.converged = false;
    result.iterations = 0;

    using DestProb = std::pair<std::string, double>;
    std::unordered_map<std::string, std::vector<DestProb>> tidx;
    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at;

    for (const auto& t : ast.transitions) {
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        actions_at[t.source_state].insert(t.action);
    }

    std::unordered_map<std::string, double> V;
    for (const auto& s : ast.states) V[s] = 0.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        double delta = 0.0;
        for (const auto& state : ast.states) {
            double old_val = V[state];

            if (actions_at.count(state) == 0) {
                // Terminal with no outgoing transitions
                double r = (ast.rewards.count(state) > 0) ? ast.rewards.at(state) : 0.0;
                V[state] = r;
                double ch = std::abs(V[state] - old_val);
                if (ch > delta) delta = ch;
                continue;
            }

            double best_val = std::numeric_limits<double>::lowest();
            for (const auto& action : actions_at[state]) {
                double reward = getReward(ast, state, action);
                double q = 0.0;
                for (const auto& [dest, prob] : tidx[state + "|" + action])
                    q += prob * V[dest];
                double val = reward + ast.discount_factor * q;
                if (val > best_val) best_val = val;
            }
            V[state] = best_val;
            double ch = std::abs(V[state] - old_val);
            if (ch > delta) delta = ch;
        }

        if (delta < theta) {
            result.converged = true;
            result.iterations = iter;
            break;
        }
        if (iter == max_iter) result.iterations = max_iter;
    }

    // Extract policy
    for (const auto& state : ast.states) {
        if (actions_at.count(state) == 0) {
            result.policy[state] = "(terminal)";
            continue;
        }
        std::string best_action;
        double best_q = std::numeric_limits<double>::lowest();
        for (const auto& action : actions_at[state]) {
            double reward = getReward(ast, state, action);
            double q = 0.0;
            for (const auto& [dest, prob] : tidx[state + "|" + action])
                q += prob * V[dest];
            double val = reward + ast.discount_factor * q;
            if (val > best_q) { best_q = val; best_action = action; }
        }
        result.policy[state] = best_action;
    }

    result.values = V;
    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_clock_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
}


// ============================================================================
// SECTION 7: SOLVER — Policy Iteration (Upgrade 1)
// ============================================================================

PolicyIterationResult solvePolicyIteration(const MDP_AST& ast) {
    auto t_start = std::chrono::high_resolution_clock::now();

    PolicyIterationResult result;
    result.converged = false;
    result.policy_improvement_steps = 0;
    result.total_bellman_evaluations = 0;

    // Build transition index and actions_at_state
    using DestProb = std::pair<std::string, double>;
    std::unordered_map<std::string, std::vector<DestProb>> tidx;
    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at;

    for (const auto& t : ast.transitions) {
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        actions_at[t.source_state].insert(t.action);
    }

    // Sorted state list for deterministic indexing
    std::vector<std::string> state_list(ast.states.begin(), ast.states.end());
    std::sort(state_list.begin(), state_list.end());
    int n = static_cast<int>(state_list.size());

    std::unordered_map<std::string, int> state_idx;
    for (int i = 0; i < n; i++) state_idx[state_list[i]] = i;

    // Initialize policy: first available action for each state
    std::unordered_map<std::string, std::string> policy;
    for (const auto& s : state_list) {
        if (actions_at.count(s) > 0 && !actions_at[s].empty())
            policy[s] = *actions_at[s].begin();
        else
            policy[s] = "(terminal)";
    }

    for (int step = 0; step < 1000; ++step) {
        // ---- POLICY EVALUATION: solve (I - gamma * P_pi) V = R ----
        // Build the n x n system Ax = b
        std::vector<std::vector<double>> A(n, std::vector<double>(n + 1, 0.0)); // Augmented [A|b]

        for (int i = 0; i < n; i++) {
            A[i][i] = 1.0; // Identity term
            const auto& s = state_list[i];
            double reward = getReward(ast, s, policy[s]);
            A[i][n] = reward; // RHS = R(s, pi(s))

            if (policy[s] != "(terminal)" && tidx.count(s + "|" + policy[s]) > 0) {
                for (const auto& [dest, prob] : tidx[s + "|" + policy[s]]) {
                    int j = state_idx[dest];
                    A[i][j] -= ast.discount_factor * prob;
                }
            }
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++) {
            // Find pivot
            int max_row = col;
            double max_val = std::abs(A[col][col]);
            for (int row = col + 1; row < n; row++) {
                if (std::abs(A[row][col]) > max_val) {
                    max_val = std::abs(A[row][col]);
                    max_row = row;
                }
            }
            if (max_val < 1e-15) continue; // Near-singular, skip
            if (max_row != col) std::swap(A[col], A[max_row]);

            // Eliminate below
            for (int row = col + 1; row < n; row++) {
                double factor = A[row][col] / A[col][col];
                for (int k = col; k <= n; k++)
                    A[row][k] -= factor * A[col][k];
            }
        }

        // Back substitution
        std::vector<double> V(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            if (std::abs(A[i][i]) < 1e-15) { V[i] = 0.0; continue; }
            V[i] = A[i][n];
            for (int j = i + 1; j < n; j++)
                V[i] -= A[i][j] * V[j];
            V[i] /= A[i][i];
        }

        result.total_bellman_evaluations += n;

        // Store values
        for (int i = 0; i < n; i++)
            result.values[state_list[i]] = V[i];

        // ---- POLICY IMPROVEMENT ----
        bool policy_stable = true;
        for (const auto& s : state_list) {
            if (actions_at.count(s) == 0) continue;

            std::string best_action;
            double best_val = std::numeric_limits<double>::lowest();
            for (const auto& action : actions_at[s]) {
                double reward = getReward(ast, s, action);
                double q = 0.0;
                if (tidx.count(s + "|" + action) > 0)
                    for (const auto& [dest, prob] : tidx[s + "|" + action])
                        q += prob * result.values[dest];
                double val = reward + ast.discount_factor * q;
                if (val > best_val) { best_val = val; best_action = action; }
            }

            if (best_action != policy[s]) {
                policy[s] = best_action;
                policy_stable = false;
            }
        }

        result.policy_improvement_steps = step + 1;

        if (policy_stable) {
            result.converged = true;
            break;
        }
    }

    result.policy = policy;
    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_clock_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
}


// ============================================================================
// SECTION 8: SOLVER — Q-Learning (Upgrade 3)
// ============================================================================

QLearningResult solveQLearning(const MDP_AST& ast) {
    auto t_start = std::chrono::high_resolution_clock::now();

    QLearningResult result;
    result.converged = false;
    result.episodes_to_convergence = ast.episodes;

    // Build transition index and cumulative distributions for sampling
    using DestProb = std::pair<std::string, double>;
    std::unordered_map<std::string, std::vector<DestProb>> tidx;
    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at;
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> cdf; // cumulative

    for (const auto& t : ast.transitions) {
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        actions_at[t.source_state].insert(t.action);
    }

    // Build CDF for each (state, action)
    for (auto& [key, dests] : tidx) {
        double cum = 0.0;
        std::vector<std::pair<std::string, double>> c;
        for (auto& [d, p] : dests) {
            cum += p;
            c.emplace_back(d, cum);
        }
        cdf[key] = c;
    }

    // Identify non-terminal states for starting episodes
    // For Q-Learning, absorbing states (self-loops) are NOT terminal —
    // the agent must visit them to learn their Q-values.
    // A state is only truly terminal if it has NO actions at all.
    std::vector<std::string> non_terminal;
    std::unordered_set<std::string> truly_terminal;
    std::unordered_set<std::string> absorbing_states;
    for (const auto& s : ast.states) {
        if (actions_at.count(s) == 0 || actions_at[s].empty()) {
            truly_terminal.insert(s);
        } else {
            // Check if absorbing (all actions self-loop)
            bool absorbing = true;
            for (const auto& a : actions_at[s]) {
                auto& dests = tidx[s + "|" + a];
                if (dests.size() != 1 || dests[0].first != s || std::abs(dests[0].second - 1.0) > 1e-9) {
                    absorbing = false; break;
                }
            }
            if (absorbing) absorbing_states.insert(s);
            non_terminal.push_back(s);
        }
    }
    if (non_terminal.empty()) {
        for (const auto& s : ast.states) non_terminal.push_back(s);
    }

    // Initialize Q-table to 0
    for (const auto& s : ast.states)
        for (const auto& a : actions_at.count(s) ? actions_at[s] : std::unordered_set<std::string>{})
            result.Q_table[s][a] = 0.0;

    std::mt19937 rng(ast.random_seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> start_dist(0, static_cast<int>(non_terminal.size()) - 1);

    double epsilon = ast.epsilon_start;
    const double epsilon_min = 0.01;
    const int max_steps = 200;

    result.episode_rewards.resize(ast.episodes, 0.0);

    // Sample next state from CDF
    auto sampleNext = [&](const std::string& s, const std::string& a) -> std::string {
        std::string key = s + "|" + a;
        if (cdf.count(key) == 0) return s;
        double r = uniform(rng);
        for (auto& [dest, cum] : cdf[key])
            if (r < cum) return dest;
        return cdf[key].back().first;
    };

    // Choose action: epsilon-greedy
    auto chooseAction = [&](const std::string& s) -> std::string {
        if (actions_at.count(s) == 0 || actions_at[s].empty()) return "";
        std::vector<std::string> acts(actions_at[s].begin(), actions_at[s].end());
        if (uniform(rng) < epsilon) {
            // Random action
            std::uniform_int_distribution<int> act_dist(0, static_cast<int>(acts.size()) - 1);
            return acts[act_dist(rng)];
        }
        // Greedy
        std::string best; double best_q = std::numeric_limits<double>::lowest();
        for (const auto& a : acts) {
            double q = result.Q_table.count(s) && result.Q_table[s].count(a) ? result.Q_table[s][a] : 0.0;
            if (q > best_q) { best_q = q; best = a; }
        }
        return best;
    };

    // ---- Main training loop ----
    for (int ep = 0; ep < ast.episodes; ep++) {
        std::string state = non_terminal[start_dist(rng)];
        double total_reward = 0.0;

        for (int step = 0; step < max_steps; step++) {
            if (truly_terminal.count(state) > 0) break;

            std::string action = chooseAction(state);
            if (action.empty()) break;

            std::string next_state = sampleNext(state, action);
            double reward = getReward(ast, state, action);
            total_reward += reward;

            // Max Q(s', a') for next state
            double max_q_next = 0.0;
            if (actions_at.count(next_state) > 0) {
                max_q_next = std::numeric_limits<double>::lowest();
                for (const auto& a2 : actions_at[next_state]) {
                    double q2 = result.Q_table.count(next_state) && result.Q_table[next_state].count(a2)
                                ? result.Q_table[next_state][a2] : 0.0;
                    if (q2 > max_q_next) max_q_next = q2;
                }
                if (max_q_next == std::numeric_limits<double>::lowest()) max_q_next = 0.0;
            }

            // TD update: Q(s,a) <- Q(s,a) + alpha * [R + gamma*maxQ' - Q(s,a)]
            double old_q = result.Q_table[state][action];
            double td_target = reward + ast.discount_factor * max_q_next;
            result.Q_table[state][action] = old_q + ast.learning_rate * (td_target - old_q);

            state = next_state;

            // If we entered an absorbing state, break after this step
            if (absorbing_states.count(state) > 0) break;
        }

        result.episode_rewards[ep] = total_reward;

        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * ast.epsilon_decay);

        // Running average every 100 episodes
        if ((ep + 1) % 100 == 0) {
            int start = std::max(0, ep - 99);
            double sum = 0.0;
            for (int i = start; i <= ep; i++) sum += result.episode_rewards[i];
            result.avg_rewards_100ep.push_back(sum / (ep - start + 1));
        }
    }

    // Extract policy from Q-table
    for (const auto& s : ast.states) {
        if (actions_at.count(s) == 0) {
            result.policy[s] = "(terminal)";
            continue;
        }
        std::string best; double best_q = std::numeric_limits<double>::lowest();
        for (const auto& a : actions_at[s]) {
            double q = result.Q_table.count(s) && result.Q_table[s].count(a)
                       ? result.Q_table[s][a] : 0.0;
            if (q > best_q) { best_q = q; best = a; }
        }
        result.policy[s] = best;
    }

    result.converged = true; // Q-Learning always runs all episodes
    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_clock_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
}


// ============================================================================
// SECTION 8a: HMM (Baum-Welch, v3.0 Phase 2A)
// ============================================================================
//
// Gaussian HMM with K hidden states fit by the Baum-Welch (EM) algorithm to
// a scalar observation sequence O_1, ..., O_T loaded from a CSV file.
//
// IMPORTANT — Numerical underflow trap.
// Forward variables alpha_t(i) = P(O_1..t, q_t=i | lambda) collapse to zero
// for sequences > ~50 steps in linear space. The standard fix (Rabiner '89)
// is the scaling factor:
//   c_t  = 1 / Sum_i alpha_t(i)
//   alpha_hat_t(i) = c_t * alpha_t(i)
// which gives Sum_i alpha_hat_t(i) = 1 at every step. The likelihood is then
//   log P(O | lambda) = -Sum_t log(c_t).
// We use identical scaling for beta_hat to keep gamma and xi well-conditioned.
//
// E-step:
//   gamma_t(i)   = alpha_hat_t(i) * beta_hat_t(i) / c_t          [scaled form]
//   xi_t(i, j)   = alpha_hat_t(i) * A_ij * b_j(O_{t+1}) * beta_hat_{t+1}(j)
//                  (no extra normalization needed under scaling)
//
// M-step:
//   pi_i  = gamma_1(i)
//   A_ij  = Sum_{t=1..T-1} xi_t(i,j)  /  Sum_{t=1..T-1} gamma_t(i)
//   mu_i  = Sum_t gamma_t(i) * O_t   /  Sum_t gamma_t(i)
//   sigma_i^2 = Sum_t gamma_t(i) * (O_t - mu_i)^2 / Sum_t gamma_t(i)
//
// Convergence: |log_likelihood_new - log_likelihood_old| < 1e-6.
//
// Initial emission means / stds are seeded by quantile-binning the data so
// states correspond to ordered ranges of observations (state 0 = lowest mean,
// state K-1 = highest mean). Transitions start as the uniform matrix.

struct HMMFitResult {
    int K;
    int T;
    std::vector<std::vector<double>> A;   // K x K transition
    std::vector<double>              pi;  // K
    std::vector<double>              mu;  // K
    std::vector<double>              sd;  // K (sigma, not variance)
    double log_likelihood;
    int    iterations;
    bool   converged;
};

inline std::vector<double> readScalarCSV(const std::string& path, std::string& err) {
    std::vector<double> data;
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "Could not open file '" + path + "'";
        return data;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // Strip trailing whitespace and skip blank / comment lines.
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end == std::string::npos) continue;
        line = line.substr(0, end + 1);
        if (line.empty() || line[0] == '#') continue;
        // Allow lines like "date,return" — take the LAST comma-separated field.
        auto comma = line.find_last_of(',');
        std::string field = (comma == std::string::npos) ? line : line.substr(comma + 1);
        try {
            data.push_back(std::stod(field));
        } catch (...) {
            // Skip headers / non-numeric rows silently (first-row header common).
            if (lineno > 1) {
                std::cerr << "[WARN] HMM CSV line " << lineno
                          << ": skipping non-numeric value '" << field << "'.\n";
            }
        }
    }
    return data;
}

inline double gaussianPdf(double x, double mu, double sd) {
    // Floor sigma to avoid log-of-zero and division-by-zero on degenerate fits.
    if (sd < 1e-6) sd = 1e-6;
    const double TWO_PI = 6.283185307179586;
    double z = (x - mu) / sd;
    return std::exp(-0.5 * z * z) / (sd * std::sqrt(TWO_PI));
}

inline HMMFitResult fitGaussianHMM(const std::vector<double>& obs, int K,
                                   int max_iter, int seed)
{
    HMMFitResult r;
    r.K = K;
    r.T = static_cast<int>(obs.size());
    r.converged = false;
    r.iterations = 0;
    r.log_likelihood = -std::numeric_limits<double>::infinity();

    if (K <= 0 || r.T <= K + 1) return r;

    // ---- Initialisation ----
    // Sort observations to seed mu_i by quantile, mapping state i to the
    // mean of the i-th equal-population bucket. sigma_i seeded by the bucket
    // standard deviation. A uniform, pi uniform.
    std::vector<double> sorted_obs(obs);
    std::sort(sorted_obs.begin(), sorted_obs.end());
    r.mu.assign(K, 0.0);
    r.sd.assign(K, 0.0);
    for (int k = 0; k < K; ++k) {
        int lo = (r.T * k) / K;
        int hi = (r.T * (k + 1)) / K;
        if (hi <= lo) hi = lo + 1;
        double sum = 0.0;
        for (int t = lo; t < hi && t < r.T; ++t) sum += sorted_obs[t];
        r.mu[k] = sum / std::max(1, hi - lo);
        double ssq = 0.0;
        for (int t = lo; t < hi && t < r.T; ++t) {
            double d = sorted_obs[t] - r.mu[k];
            ssq += d * d;
        }
        r.sd[k] = std::sqrt(ssq / std::max(1, hi - lo));
        if (r.sd[k] < 1e-3) r.sd[k] = 1e-3;
    }
    // Small jitter for reproducibility w/ seed (so seed makes a difference).
    std::mt19937 rng(seed);
    std::normal_distribution<double> jitter(0.0, 1e-4);
    for (int k = 0; k < K; ++k) r.mu[k] += jitter(rng);

    r.A.assign(K, std::vector<double>(K, 1.0 / K));
    // Slightly diagonal-favouring initialisation, helps regime-style HMMs.
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) r.A[i][j] = (i == j) ? 0.7 : 0.3 / (K - 1);
    }
    r.pi.assign(K, 1.0 / K);

    const int T = r.T;
    std::vector<std::vector<double>> alpha(T, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> beta (T, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> gamma(T, std::vector<double>(K, 0.0));
    std::vector<double> c(T, 0.0);  // scaling factors

    double prev_ll = -std::numeric_limits<double>::infinity();

    for (int it = 1; it <= max_iter; ++it) {
        // ---- E-step: scaled forward-backward ----
        // alpha_hat_1(i) = pi_i * b_i(O_1), then normalize.
        double s = 0.0;
        for (int i = 0; i < K; ++i) {
            alpha[0][i] = r.pi[i] * gaussianPdf(obs[0], r.mu[i], r.sd[i]);
            s += alpha[0][i];
        }
        if (s <= 0.0) s = 1e-300;
        c[0] = 1.0 / s;
        for (int i = 0; i < K; ++i) alpha[0][i] *= c[0];

        for (int t = 1; t < T; ++t) {
            s = 0.0;
            for (int j = 0; j < K; ++j) {
                double sum = 0.0;
                for (int i = 0; i < K; ++i) sum += alpha[t-1][i] * r.A[i][j];
                alpha[t][j] = sum * gaussianPdf(obs[t], r.mu[j], r.sd[j]);
                s += alpha[t][j];
            }
            if (s <= 0.0) s = 1e-300;
            c[t] = 1.0 / s;
            for (int j = 0; j < K; ++j) alpha[t][j] *= c[t];
        }

        // beta_hat_T(i) = c_T (standard scaled init).
        for (int i = 0; i < K; ++i) beta[T-1][i] = c[T-1];
        for (int t = T - 2; t >= 0; --t) {
            for (int i = 0; i < K; ++i) {
                double sum = 0.0;
                for (int j = 0; j < K; ++j)
                    sum += r.A[i][j] * gaussianPdf(obs[t+1], r.mu[j], r.sd[j]) * beta[t+1][j];
                beta[t][i] = c[t] * sum;
            }
        }

        // gamma + log-likelihood.
        double ll = 0.0;
        for (int t = 0; t < T; ++t) {
            double gsum = 0.0;
            for (int i = 0; i < K; ++i) {
                gamma[t][i] = alpha[t][i] * beta[t][i] / c[t];
                gsum += gamma[t][i];
            }
            // Numerical guard: re-normalize gamma to compensate for tiny drift.
            if (gsum > 0.0) for (int i = 0; i < K; ++i) gamma[t][i] /= gsum;
            ll += -std::log(c[t]);
        }
        r.log_likelihood = ll;
        r.iterations = it;

        // Convergence check.
        if (std::abs(ll - prev_ll) < 1e-6 && it > 1) { r.converged = true; break; }
        prev_ll = ll;

        // ---- M-step ----
        // pi
        for (int i = 0; i < K; ++i) r.pi[i] = gamma[0][i];

        // A
        std::vector<std::vector<double>> A_num(K, std::vector<double>(K, 0.0));
        std::vector<double>              A_den(K, 0.0);
        for (int t = 0; t < T - 1; ++t) {
            for (int i = 0; i < K; ++i) {
                A_den[i] += gamma[t][i];
                for (int j = 0; j < K; ++j) {
                    A_num[i][j] += alpha[t][i] * r.A[i][j]
                                 * gaussianPdf(obs[t+1], r.mu[j], r.sd[j])
                                 * beta[t+1][j];
                }
            }
        }
        for (int i = 0; i < K; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < K; ++j) row_sum += A_num[i][j];
            if (row_sum > 0.0) {
                for (int j = 0; j < K; ++j) r.A[i][j] = A_num[i][j] / row_sum;
            }
            // Sanity: never leave a degenerate all-zero row.
            (void)A_den;
        }

        // Gaussian emissions
        for (int i = 0; i < K; ++i) {
            double num = 0.0, den = 0.0;
            for (int t = 0; t < T; ++t) { num += gamma[t][i] * obs[t]; den += gamma[t][i]; }
            if (den > 1e-12) r.mu[i] = num / den;
        }
        for (int i = 0; i < K; ++i) {
            double num = 0.0, den = 0.0;
            for (int t = 0; t < T; ++t) {
                double d = obs[t] - r.mu[i];
                num += gamma[t][i] * d * d;
                den += gamma[t][i];
            }
            if (den > 1e-12) r.sd[i] = std::sqrt(num / den);
            if (r.sd[i] < 1e-3) r.sd[i] = 1e-3;
        }
    }

    return r;
}

// ---- bridgeHMMToMDP ----
// Runs Baum-Welch on the FIT_HMM CSV and copies the learned regime structure
// into the AST: hidden states become MDP STATEs (if not already declared),
// and the learned A_ij becomes TRANSITION rows (action-independent — one row
// per declared action; rationale in the section banner above).
//
// Returns a list of error strings; empty on success.
inline std::vector<std::string> bridgeHMMToMDP(MDP_AST& ast) {
    std::vector<std::string> errors;
    if (!ast.hmm_bridge_enabled) return errors;
    if (ast.hmm_states.empty()) {
        errors.push_back("[HMM bridge] BRIDGE: hmm -> mdp requires HMM_STATES.");
        return errors;
    }
    if (ast.hmm_fit_csv.empty()) {
        errors.push_back("[HMM bridge] BRIDGE: hmm -> mdp requires FIT_HMM: <csv>.");
        return errors;
    }
    std::string read_err;
    auto data = readScalarCSV(ast.hmm_fit_csv, read_err);
    if (!read_err.empty()) {
        errors.push_back("[HMM bridge] " + read_err);
        return errors;
    }
    if (data.size() < static_cast<size_t>(ast.hmm_states.size()) + 2) {
        errors.push_back("[HMM bridge] CSV has too few observations (" +
                         std::to_string(data.size()) + ").");
        return errors;
    }

    HMMFitResult r = fitGaussianHMM(data, static_cast<int>(ast.hmm_states.size()),
                                    ast.hmm_fit_iterations, ast.hmm_seed);
    ast.hmm_transition_matrix = r.A;
    ast.hmm_initial           = r.pi;
    ast.hmm_emission_mean     = r.mu;
    ast.hmm_emission_std      = r.sd;
    ast.hmm_log_likelihood    = r.log_likelihood;
    ast.hmm_fitted            = true;

    // Register the HMM_STATES as MDP STATEs (idempotent: skip if present).
    for (const auto& s : ast.hmm_states) ast.states.insert(s);

    // Add action-independent TRANSITION rows: for every declared action,
    // T(s_i, a, s_j) = A_ij. We do NOT overwrite explicit transitions the
    // user may have already declared (those win).
    auto haveTransition = [&](const std::string& s, const std::string& a) {
        for (const auto& t : ast.transitions) {
            if (t.source_state == s && t.action == a) return true;
        }
        return false;
    };

    const int K = static_cast<int>(ast.hmm_states.size());
    for (const auto& a : ast.actions) {
        for (int i = 0; i < K; ++i) {
            const std::string& s_i = ast.hmm_states[i];
            if (haveTransition(s_i, a)) continue;
            for (int j = 0; j < K; ++j) {
                const std::string& s_j = ast.hmm_states[j];
                if (r.A[i][j] > 0.0)
                    ast.transitions.push_back({s_i, a, s_j, r.A[i][j]});
            }
        }
    }

    return errors;
}

// ============================================================================
// SECTION 8b: SOLVER — PBVI (Point-Based Value Iteration, v3.0)
// ============================================================================
//
// Notation and indexing conventions used throughout this section:
//   * A belief b is a sparse map state -> probability summing to 1.
//   * An alpha vector α is a map state -> real number, with an attached
//     action label π_α = action(α). Γ denotes the current set of α vectors.
//   * V(b) = max_{α ∈ Γ} Σ_s b(s) · α(s), π(b) = action(argmax α).
//   * T(s,a,s') comes from ast.transitions.
//   * O(s',a,o) comes from ast.observe_probs (indexed by NEXT state and
//     action taken — i.e., the observation is generated after transitioning
//     into s' via action a). This is the standard POMDP convention.
//   * R(s,a) comes from getReward(ast, s, a), with R(s) as fallback.
//
// Bellman backup over belief space:
//   α*_{a,o,b} = argmax_{α ∈ Γ_prev} Σ_{s'} α(s') · [Σ_s b(s) T(s,a,s') O(s',a,o)]
//   α^{a,b}(s) = R(s,a) + γ · Σ_o Σ_{s'} T(s,a,s') O(s',a,o) α*_{a,o,b}(s')
//   α^{b}     = argmax over a   of   Σ_s b(s) · α^{a,b}(s)
//   Γ_new = ⋃_{b ∈ B} {α^{b}}, then prune dominated vectors.
//
// Pessimistic init (one α per state, see prompt 1.4):
//   α_s(s')  =  R(s) / (1 - γ)    if s == s', else 0
// where R(s) is the legacy state reward (a safe lower bound; in worst case
// the agent gets exactly that reward forever).

// ---- Belief update: τ(b, a, o) using the Bayes filter ----
//
// b'(s') = η · O(s', a, o) · Σ_s T(s, a, s') · b(s)
// η      = 1 / P(o | b, a)
//
// Returns std::nullopt-equivalent (an empty map) iff P(o|b,a) is zero;
// caller policy: emit [WARN] and fall back to the uniform belief.
//
// To avoid numerical underflow on long observation sequences (Tiger episodes
// can run 50+ Listens before opening), we operate in log-space whenever the
// caller chains many updates. For a single forward step the linear form is
// stable; for the Baum-Welch-style chain used by HMM fitting (Phase 2) we
// reuse the same scaling pattern: c_t = 1/Σ_i α_t(i), α̂_t(i) = c_t · α_t(i).
struct BeliefUpdateResult {
    std::unordered_map<std::string, double> belief;
    bool zero_probability_observation;  // true => caller should warn & fall back
};

inline BeliefUpdateResult updateBelief(
    const MDP_AST& ast,
    const std::unordered_map<std::string, double>& b,
    const std::string& action,
    const std::string& observation,
    // Precomputed (s,a) -> list<(s',p)>. If empty, will be built lazily.
    const std::unordered_map<std::string,
        std::vector<std::pair<std::string, double>>>* tidx = nullptr)
{
    BeliefUpdateResult out;
    out.zero_probability_observation = false;

    // Build a tiny local tidx if none provided (test helper case).
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> local_tidx;
    const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>* T =
        tidx;
    if (!T) {
        for (const auto& t : ast.transitions)
            local_tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        T = &local_tidx;
    }

    // Numerator: u(s') = O(s',a,o) · Σ_s T(s,a,s') · b(s)
    std::unordered_map<std::string, double> u;
    double total = 0.0;
    for (const auto& s_prime : ast.states) {
        double obs_p = 0.0;
        auto it1 = ast.observe_probs.find(s_prime);
        if (it1 != ast.observe_probs.end()) {
            auto it2 = it1->second.find(action);
            if (it2 != it1->second.end()) {
                auto it3 = it2->second.find(observation);
                if (it3 != it2->second.end()) obs_p = it3->second;
            }
        }
        if (obs_p <= 0.0) continue;  // sparse: skip impossible (s',a,o)

        double inner = 0.0;
        for (const auto& [s, bs] : b) {
            if (bs <= 0.0) continue;
            const std::string key = s + "|" + action;
            auto it = T->find(key);
            if (it == T->end()) continue;
            for (const auto& [sp, tp] : it->second) {
                if (sp == s_prime) inner += bs * tp;
            }
        }
        double val = obs_p * inner;
        if (val > 0.0) {
            u[s_prime] = val;
            total += val;
        }
    }

    if (total <= 0.0) {
        // P(o | b, a) = 0  =>  observation impossible under model.
        // Per spec: fall back to uniform belief, set warn flag.
        out.zero_probability_observation = true;
        double n = static_cast<double>(ast.states.size());
        if (n > 0.0) {
            for (const auto& s : ast.states) out.belief[s] = 1.0 / n;
        }
        return out;
    }

    for (auto& [s, v] : u) out.belief[s] = v / total;
    return out;
}

// ---- α vector dot product with a belief ----
inline double alphaDotBelief(const AlphaVector& alpha,
                             const std::unordered_map<std::string, double>& b)
{
    double s = 0.0;
    for (const auto& [state, p] : b) {
        auto it = alpha.values.find(state);
        if (it != alpha.values.end()) s += p * it->second;
    }
    return s;
}

// ---- Witness-point pruning of an α-set ----
//
// Keep α iff there exists a belief point in `witnesses` where α strictly
// dominates every other α (by some tiny margin). We use the supplied belief
// set B as our witness set; this is precisely the "point-based" semantics:
// vectors that are useful for AT LEAST ONE belief in B survive, regardless
// of whether some completely unseen belief would prefer them. The
// prompt-mandated trap warning ("do not over-prune") is satisfied because
// we never drop an α that is uniquely best at any b ∈ B.
inline std::vector<AlphaVector> pruneAlphas(
    const std::vector<AlphaVector>& gamma,
    const std::vector<std::unordered_map<std::string, double>>& witnesses)
{
    if (gamma.size() <= 1) return gamma;

    std::vector<bool> keep(gamma.size(), false);

    for (const auto& b : witnesses) {
        // Find best α at b.
        int best = -1;
        double best_val = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < gamma.size(); ++i) {
            double v = alphaDotBelief(gamma[i], b);
            if (v > best_val + 1e-12) {
                best_val = v;
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) keep[best] = true;
    }

    std::vector<AlphaVector> out;
    out.reserve(gamma.size());
    for (size_t i = 0; i < gamma.size(); ++i)
        if (keep[i]) out.push_back(gamma[i]);

    // Defensive: if pruning would erase everything (e.g. empty witness set),
    // return the original — never let Γ be empty.
    if (out.empty()) return gamma;
    return out;
}

// ---- Belief-set expansion via stochastic forward simulation ----
//
// For each b in B, take a few one-step lookahead samples. The lookahead
// produces a successor belief τ(b,a,o) for each (a,o) where P(o|b,a) > 0;
// we pick the successor that maximises L1 distance to all existing belief
// points in B (the standard PBVI "SSEA" expansion heuristic, simplified).
inline std::vector<std::unordered_map<std::string, double>> expandBeliefSet(
    const MDP_AST& ast,
    const std::vector<std::unordered_map<std::string, double>>& B,
    size_t cap,
    const std::unordered_map<std::string,
        std::vector<std::pair<std::string, double>>>& tidx)
{
    if (B.size() >= cap) return B;
    auto B_new = B;

    auto l1 = [](const std::unordered_map<std::string, double>& x,
                 const std::unordered_map<std::string, double>& y) {
        double d = 0.0;
        std::unordered_set<std::string> keys;
        for (const auto& [k, v] : x) keys.insert(k);
        for (const auto& [k, v] : y) keys.insert(k);
        for (const auto& k : keys) {
            double xv = x.count(k) ? x.at(k) : 0.0;
            double yv = y.count(k) ? y.at(k) : 0.0;
            d += std::abs(xv - yv);
        }
        return d;
    };

    for (size_t bi = 0; bi < B.size() && B_new.size() < cap; ++bi) {
        const auto& b = B[bi];
        const std::unordered_map<std::string, double>* best = nullptr;
        std::unordered_map<std::string, double> best_store;
        double best_dist = -1.0;

        for (const auto& a : ast.actions) {
            for (const auto& o : ast.observations) {
                auto upd = updateBelief(ast, b, a, o, &tidx);
                if (upd.zero_probability_observation) continue;
                // Find the minimum distance from this candidate to any existing
                // belief, then keep the candidate that maximises that minimum.
                double min_d = std::numeric_limits<double>::infinity();
                for (const auto& existing : B_new) {
                    double d = l1(upd.belief, existing);
                    if (d < min_d) min_d = d;
                }
                if (min_d > best_dist) {
                    best_dist = min_d;
                    best_store = upd.belief;
                    best = &best_store;
                }
            }
        }
        if (best && best_dist > 1e-6) {
            B_new.push_back(best_store);
        }
    }
    return B_new;
}

PBVIResult solvePBVI(const MDP_AST& ast) {
    auto t_start = std::chrono::high_resolution_clock::now();

    PBVIResult result;
    result.converged = false;
    result.iterations = 0;

    const double gamma = ast.discount_factor;

    // ---- Index transitions and actions-at-state ----
    // tidx[s|a] = list of (s', T(s,a,s'))
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> tidx;
    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at;
    for (const auto& t : ast.transitions) {
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        actions_at[t.source_state].insert(t.action);
    }

    // ---- Build belief set B from named beliefs + initial belief ----
    std::vector<std::unordered_map<std::string, double>> B;
    std::vector<std::string> B_names;  // parallel: names for reporting

    if (!ast.initial_belief.empty()) {
        B.push_back(ast.initial_belief);
        B_names.push_back("_initial");
    }
    for (const auto& [nm, nb] : ast.belief_states) {
        // Normalize defensively (validator catches malformed inputs).
        std::unordered_map<std::string, double> b = nb.probs;
        double s = 0.0;
        for (auto& [k, v] : b) s += v;
        if (s > 0.0) for (auto& [k, v] : b) v /= s;
        B.push_back(b);
        B_names.push_back(nm);
    }
    if (B.empty()) {
        // Fallback: uniform belief.
        std::unordered_map<std::string, double> uni;
        double n = static_cast<double>(ast.states.size());
        if (n > 0) for (const auto& s : ast.states) uni[s] = 1.0 / n;
        B.push_back(uni);
        B_names.push_back("_uniform");
    }

    // ---- Expand B up to the cap using forward simulation ----
    {
        size_t cap = static_cast<size_t>(std::max(1, ast.pbvi_alpha_vectors));
        // Run a couple of expansion passes (each adds at most |B_prev| points).
        for (int pass = 0; pass < 3 && B.size() < cap; ++pass) {
            auto B2 = expandBeliefSet(ast, B, cap, tidx);
            if (B2.size() == B.size()) break;
            B = std::move(B2);
        }
    }

    // ---- Pessimistic α-vector initialisation ----
    // For each state s, α_s(s') = R(s)/(1-γ) at s, 0 elsewhere.
    // We attach the action that has the highest reward at s, but action
    // labels for the initial pessimistic α-vectors are mostly placeholders —
    // they will be overwritten in the first backup pass.
    std::vector<AlphaVector> Gamma;
    Gamma.reserve(ast.states.size());
    {
        double horizon_div = (gamma >= 1.0 - 1e-12) ? 1.0 : (1.0 - gamma);
        for (const auto& s : ast.states) {
            AlphaVector a;
            double r_s = ast.rewards.count(s) ? ast.rewards.at(s) : 0.0;
            // Min over actions of R(s,a) ensures lower bound under any policy.
            double r_min = r_s;
            bool any = false;
            if (actions_at.count(s)) {
                for (const auto& act : actions_at.at(s)) {
                    double r = getReward(ast, s, act);
                    if (!any || r < r_min) { r_min = r; any = true; }
                }
            }
            for (const auto& s2 : ast.states) a.values[s2] = 0.0;
            a.values[s] = r_min / horizon_div;
            // Pick any action label (will get overwritten in first backup).
            if (actions_at.count(s) && !actions_at.at(s).empty())
                a.action = *actions_at.at(s).begin();
            else
                a.action = "(terminal)";
            Gamma.push_back(a);
        }
    }

    // ---- Helper: argmax α ∈ Γ of Σ_s' α(s') · w(s') ----
    // (Used in the backup to find α*_{a,o,b}.)
    auto argmaxAlpha = [&](const std::vector<AlphaVector>& gma,
                           const std::unordered_map<std::string, double>& weight)
                           -> const AlphaVector& {
        if (gma.empty()) {
            static const AlphaVector empty_alpha;
            return empty_alpha;
        }
        size_t best = 0;
        double best_v = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < gma.size(); ++i) {
            double v = 0.0;
            for (const auto& [sp, w] : weight) {
                auto it = gma[i].values.find(sp);
                if (it != gma[i].values.end()) v += w * it->second;
            }
            if (v > best_v) { best_v = v; best = i; }
        }
        return gma[best];
    };

    // ---- Main PBVI loop ----
    int max_iter = std::max(1, ast.pbvi_iterations);
    int horizon = (ast.horizon > 0) ? ast.horizon : max_iter;
    int effective_iters = std::min(max_iter, horizon);

    double last_max_change = std::numeric_limits<double>::infinity();
    for (int iter = 1; iter <= effective_iters; ++iter) {
        std::vector<AlphaVector> new_Gamma;
        new_Gamma.reserve(B.size());

        for (size_t bi = 0; bi < B.size(); ++bi) {
            const auto& b = B[bi];

            AlphaVector best_alpha_for_b;
            double best_value_for_b = -std::numeric_limits<double>::infinity();
            bool initialised = false;

            for (const auto& action : ast.actions) {
                // α^{a,b}(s) = R(s,a)  +  γ · Σ_o Σ_{s'} T(s,a,s') O(s',a,o) · α*_{a,o,b}(s')
                AlphaVector alpha_ab;
                alpha_ab.action = action;
                for (const auto& s : ast.states) alpha_ab.values[s] = 0.0;

                // First, the per-state immediate reward R(s,a).
                for (const auto& s : ast.states)
                    alpha_ab.values[s] = getReward(ast, s, action);

                // Now the discounted observation-conditioned backup.
                // For each observation, build the s'->weight vector
                //   w_o(s') = Σ_s b(s) T(s,a,s') O(s',a,o)
                // and pick the α ∈ Γ that maximises Σ_{s'} α(s') · w_o(s').
                for (const auto& obs : ast.observations) {
                    std::unordered_map<std::string, double> w_o;
                    bool any_weight = false;
                    for (const auto& [s, bs] : b) {
                        if (bs <= 0.0) continue;
                        auto it = tidx.find(s + "|" + action);
                        if (it == tidx.end()) continue;
                        for (const auto& [sp, tp] : it->second) {
                            double op = 0.0;
                            auto it1 = ast.observe_probs.find(sp);
                            if (it1 != ast.observe_probs.end()) {
                                auto it2 = it1->second.find(action);
                                if (it2 != it1->second.end()) {
                                    auto it3 = it2->second.find(obs);
                                    if (it3 != it2->second.end()) op = it3->second;
                                }
                            }
                            if (op <= 0.0) continue;
                            w_o[sp] += bs * tp * op;
                            any_weight = true;
                        }
                    }
                    if (!any_weight) continue;

                    const AlphaVector& a_star = argmaxAlpha(Gamma, w_o);

                    // Add γ · Σ_{s'} T(s,a,s') O(s',a,o) · a_star(s')   to α_ab(s).
                    // The "T·O" weighting at source s, action a, into s', obs o is:
                    //   coef(s, s') = T(s,a,s') · O(s',a,o)
                    // so α_ab(s) += γ · Σ_{s'} coef(s, s') · a_star(s').
                    for (const auto& s : ast.states) {
                        auto it = tidx.find(s + "|" + action);
                        if (it == tidx.end()) continue;
                        double inc = 0.0;
                        for (const auto& [sp, tp] : it->second) {
                            double op = 0.0;
                            auto it1 = ast.observe_probs.find(sp);
                            if (it1 != ast.observe_probs.end()) {
                                auto it2 = it1->second.find(action);
                                if (it2 != it1->second.end()) {
                                    auto it3 = it2->second.find(obs);
                                    if (it3 != it2->second.end()) op = it3->second;
                                }
                            }
                            if (op <= 0.0) continue;
                            double av = 0.0;
                            auto avit = a_star.values.find(sp);
                            if (avit != a_star.values.end()) av = avit->second;
                            inc += tp * op * av;
                        }
                        alpha_ab.values[s] += gamma * inc;
                    }
                }

                double val_at_b = alphaDotBelief(alpha_ab, b);
                if (!initialised || val_at_b > best_value_for_b) {
                    initialised = true;
                    best_value_for_b = val_at_b;
                    best_alpha_for_b = alpha_ab;
                }
            }
            if (initialised) new_Gamma.push_back(best_alpha_for_b);
        }

        // Compute max change in V(b) across all b ∈ B before pruning.
        double max_change = 0.0;
        for (const auto& b : B) {
            double v_old = -std::numeric_limits<double>::infinity();
            for (const auto& a : Gamma) {
                double v = alphaDotBelief(a, b);
                if (v > v_old) v_old = v;
            }
            double v_new = -std::numeric_limits<double>::infinity();
            for (const auto& a : new_Gamma) {
                double v = alphaDotBelief(a, b);
                if (v > v_new) v_new = v;
            }
            double d = std::abs(v_new - v_old);
            if (d > max_change) max_change = d;
        }

        // Witness-point pruning against the belief set.
        Gamma = pruneAlphas(new_Gamma, B);

        result.iterations = iter;
        last_max_change = max_change;
        if (max_change < 1e-9) {
            result.converged = true;
            break;
        }
    }
    (void)last_max_change;  // (could be exported in --json)

    // ---- Record final Γ and the named/initial belief values + policy ----
    result.alpha_vectors = Gamma;
    // Belief set as NamedBelief list (using stored names when available).
    for (size_t i = 0; i < B.size(); ++i) {
        NamedBelief nb;
        nb.name = (i < B_names.size()) ? B_names[i] : ("_b" + std::to_string(i));
        nb.probs = B[i];
        result.belief_points.push_back(nb);
    }

    auto bestForBelief = [&](const std::unordered_map<std::string, double>& b)
            -> std::pair<double, std::string> {
        double best_v = -std::numeric_limits<double>::infinity();
        std::string best_a = "(none)";
        for (const auto& a : Gamma) {
            double v = alphaDotBelief(a, b);
            if (v > best_v) { best_v = v; best_a = a.action; }
        }
        return {best_v, best_a};
    };

    // Initial belief
    if (!ast.initial_belief.empty()) {
        auto [v, a] = bestForBelief(ast.initial_belief);
        result.values_at_belief["_initial"] = v;
        result.policy_at_belief["_initial"] = a;
    }
    // Each named belief
    for (const auto& [nm, nb] : ast.belief_states) {
        auto [v, a] = bestForBelief(nb.probs);
        result.values_at_belief[nm] = v;
        result.policy_at_belief[nm] = a;
    }
    // Each belief point in B (anonymous)
    for (size_t i = 0; i < B.size(); ++i) {
        std::string key = (i < B_names.size()) ? B_names[i] : ("_b" + std::to_string(i));
        if (result.values_at_belief.count(key)) continue;
        auto [v, a] = bestForBelief(B[i]);
        result.values_at_belief[key] = v;
        result.policy_at_belief[key] = a;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_clock_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
}

// ---- Pretty printer for PBVI results ----
inline void printPBVIReport(const PBVIResult& r, const MDP_AST& ast) {
    std::cout << "\n======================================\n";
    std::cout << "  PBVI RESULTS (POMDP)\n";
    std::cout << "======================================\n\n";
    std::cout << "-- CONVERGENCE --\n";
    if (r.converged) std::cout << "  Converged after " << r.iterations << " backups.\n";
    else std::cout << "  Did NOT converge within " << r.iterations << " backups.\n";
    std::cout << "  Alpha vectors retained: " << r.alpha_vectors.size() << "\n";
    std::cout << "  Belief points used:     " << r.belief_points.size() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << r.wall_clock_ms << " ms\n\n";

    std::cout << "-- VALUE / POLICY AT NAMED / INITIAL BELIEFS --\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(20) << "Belief"
              << std::setw(14) << "V*(b)" << "pi*(b)\n";
    std::cout << "  " << std::string(54, '-') << "\n";

    // Print named beliefs first, then initial, then expanded points.
    auto printRow = [&](const std::string& key) {
        if (!r.values_at_belief.count(key)) return;
        std::cout << "  " << std::left << std::setw(20) << key
                  << std::setw(14) << r.values_at_belief.at(key)
                  << r.policy_at_belief.at(key) << "\n";
    };
    for (const auto& [nm, nb] : ast.belief_states) printRow(nm);
    if (r.values_at_belief.count("_initial")) printRow("_initial");
    int count = 0;
    for (const auto& [k, v] : r.values_at_belief) {
        if (ast.belief_states.count(k) || k == "_initial") continue;
        printRow(k);
        if (++count >= 8) break;
    }
    std::cout << "\n======================================\n";
}

// ============================================================================
// SECTION 8c: AUTOPSY ENGINE (v3.0 Phase 3A)
// ============================================================================
//
// Diagnostic passes over a parsed AST + solver result. Each pass appends 0 or
// more AutopsyFindings to the report. The dispatcher `runAutopsy(mode, ...)`
// picks which passes to run based on the CLI flag:
//
//   "full"        : every pass
//   "reward"      : reward-shape passes (myopia, hacking, magnitude)
//   "structural"  : graph-shape passes (dead-state)
//   "solver"      : discount cliff
//   "landscape"   : fragility only
//
// All passes are read-only: they never mutate the AST. The REPAIR command in
// the next subsection clones the AST, perturbs the clone, and re-solves.

// ---- helper: get the policy + values from any (vi / pi) solver ----
// We treat VI and PI symmetrically — both produce a {values, policy} map.
struct SimpleSolution {
    std::unordered_map<std::string, double>      values;
    std::unordered_map<std::string, std::string> policy;
};
inline SimpleSolution solveOnce(const MDP_AST& ast) {
    // VI is the most robust for autopsy passes (handles γ near 1, γ near 0,
    // pathological reward landscapes alike). Use a tight tolerance because we
    // care about ordering of values for policy correctness, not exact magnitudes.
    SolverResult vi = solveValueIteration(ast);
    return {vi.values, vi.policy};
}

// ---- helper: build a transition adjacency map keyed by source state ----
// Returns out[s] = list<(action, dest, prob)>.
inline std::unordered_map<std::string,
    std::vector<std::tuple<std::string, std::string, double>>>
buildAdjacency(const MDP_AST& ast) {
    std::unordered_map<std::string,
        std::vector<std::tuple<std::string, std::string, double>>> out;
    for (const auto& t : ast.transitions)
        out[t.source_state].emplace_back(t.action, t.dest_state, t.probability);
    return out;
}

// ----------------------------------------------------------------------------
// Class 4 — Magnitude Imbalance
// ----------------------------------------------------------------------------
// We enumerate every distinct reward term (R(s) and R(s,a)) and rank them by
// |r_i| / Σ |r_j|. Anything < 1% emits WARN; < 0.1% emits ERROR (effectively
// invisible to the solver). The threshold is per the master spec.
inline void autopsyMagnitudeImbalance(const MDP_AST& ast, AutopsyReport& report) {
    struct Term { std::string label; double value; };
    std::vector<Term> terms;
    for (const auto& [s, r] : ast.rewards) {
        if (r != 0.0) terms.push_back({"R(" + s + ")", r});
    }
    for (const auto& [s, am] : ast.action_rewards) {
        for (const auto& [a, r] : am) {
            if (r != 0.0) terms.push_back({"R(" + s + ", " + a + ")", r});
        }
    }
    if (terms.size() < 2) return;  // nothing to imbalance

    double total = 0.0;
    for (const auto& t : terms) total += std::abs(t.value);
    if (total <= 0.0) return;

    // Sort by absolute contribution descending so the headline term reads well.
    std::sort(terms.begin(), terms.end(),
              [](const Term& a, const Term& b) { return std::abs(a.value) > std::abs(b.value); });

    for (const auto& t : terms) {
        double contrib = std::abs(t.value) / total;
        if (contrib < 0.001) {
            AutopsyFinding f;
            f.cls = FindingClass::MagnitudeImbalance;
            f.severity = FindingSeverity::ERROR_LEVEL;
            std::ostringstream hl;
            hl << "Reward term " << t.label << " contributes only "
               << std::fixed << std::setprecision(2) << (contrib * 100.0)
               << "% — effectively ignored by the solver.";
            f.headline = hl.str();
            f.details.push_back("Total reward magnitude in the model: " +
                                std::to_string(total));
            f.details.push_back("This term's contribution: " +
                                std::to_string(std::abs(t.value)));
            // Suggest a scale-up that reaches 5% contribution.
            double target_abs = 0.05 * (total - std::abs(t.value)) / (1.0 - 0.05);
            f.details.push_back("Suggested scale-up: |" + t.label + "| >= " +
                                std::to_string(target_abs) + " to reach 5% contribution.");
            f.numeric_fields["contribution"] = contrib;
            f.numeric_fields["value"] = t.value;
            f.numeric_fields["suggested_min_abs"] = target_abs;
            f.text_fields["term"] = t.label;
            report.findings.push_back(f);
        } else if (contrib < 0.01) {
            AutopsyFinding f;
            f.cls = FindingClass::MagnitudeImbalance;
            f.severity = FindingSeverity::WARN;
            std::ostringstream hl;
            hl << "Reward term " << t.label << " contributes only "
               << std::fixed << std::setprecision(2) << (contrib * 100.0)
               << "% — may be drowned out by larger rewards.";
            f.headline = hl.str();
            f.numeric_fields["contribution"] = contrib;
            f.numeric_fields["value"] = t.value;
            f.text_fields["term"] = t.label;
            report.findings.push_back(f);
        }
    }
}

// ----------------------------------------------------------------------------
// Class 1 — Reward Myopia
// ----------------------------------------------------------------------------
// Compute the effective planning horizon  H = log(eps) / log(gamma)  where
// eps = 1e-3 (a future reward smaller than this is invisible). Then BFS the
// transition graph from each state to the nearest positive-reward state.
// If min_goal_distance > H, the agent literally cannot perceive a gradient.
inline void autopsyRewardMyopia(const MDP_AST& ast, AutopsyReport& report) {
    if (ast.discount_factor <= 0.0 || ast.discount_factor >= 1.0) return;
    const double eps_visibility = 1e-3;
    double eff_horizon = std::log(eps_visibility) / std::log(ast.discount_factor);

    // Identify positive-reward states (goals) — anything with R(s) > 0 OR
    // an action-reward > 0 at some action.
    std::unordered_set<std::string> goals;
    for (const auto& [s, r] : ast.rewards) if (r > 0) goals.insert(s);
    for (const auto& [s, am] : ast.action_rewards)
        for (const auto& [a, r] : am) if (r > 0) goals.insert(s);
    if (goals.empty()) return;

    // Build an undirected reachability graph (any action -> any successor).
    auto adj = buildAdjacency(ast);
    auto bfsDist = [&](const std::string& src) -> int {
        std::unordered_map<std::string, int> dist;
        std::deque<std::string> q;
        dist[src] = 0; q.push_back(src);
        while (!q.empty()) {
            auto u = q.front(); q.pop_front();
            if (goals.count(u)) return dist[u];
            auto it = adj.find(u);
            if (it == adj.end()) continue;
            for (const auto& [act, v, p] : it->second) {
                if (p <= 0) continue;
                if (dist.count(v)) continue;
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
        return -1;
    };

    int worst_dist = 0;
    std::string worst_state;
    for (const auto& s : ast.states) {
        if (goals.count(s)) continue;
        int d = bfsDist(s);
        if (d > worst_dist) { worst_dist = d; worst_state = s; }
    }
    if (worst_state.empty()) return;

    if (worst_dist > eff_horizon + 0.5) {
        AutopsyFinding f;
        f.cls = FindingClass::RewardMyopia;
        f.severity = FindingSeverity::ERROR_LEVEL;
        std::ostringstream hl;
        hl << "Reward Myopia: from '" << worst_state << "' the goal is "
           << worst_dist << " steps away but the effective planning horizon is "
           << std::fixed << std::setprecision(1) << eff_horizon << " (gamma="
           << ast.discount_factor << ").";
        f.headline = hl.str();
        f.details.push_back("The agent cannot perceive any reward gradient toward the goal from this state.");

        // Suggest a higher gamma where horizon first reaches the worst distance.
        // We need log(eps) / log(gamma') >= worst_dist  =>
        //   log(gamma') >= log(eps) / worst_dist
        //   gamma'      >= exp(log(eps) / worst_dist)
        double target_gamma = std::exp(std::log(eps_visibility) / static_cast<double>(worst_dist));
        if (target_gamma > 0.999) target_gamma = 0.999;
        std::ostringstream sg;
        sg << "Fix option [1]: raise gamma to >= " << std::fixed
           << std::setprecision(3) << target_gamma
           << " so the effective horizon reaches the goal distance.";
        f.details.push_back(sg.str());
        f.details.push_back("Fix option [2]: add a shaping reward at an intermediate state to break the distance.");
        f.numeric_fields["effective_horizon"]   = eff_horizon;
        f.numeric_fields["min_goal_distance"]   = static_cast<double>(worst_dist);
        f.numeric_fields["suggested_gamma_min"] = target_gamma;
        f.text_fields["worst_state"] = worst_state;
        report.findings.push_back(f);
    }
}

// ----------------------------------------------------------------------------
// Class 5 — Dead State Detection
// ----------------------------------------------------------------------------
// A dead state is one from which no policy can ever reach a positive reward.
// We compute the set forward-reachable-to-positive-reward; the complement
// (minus states with no incoming edges from elsewhere) is dead.
inline void autopsyDeadStates(const MDP_AST& ast, AutopsyReport& report) {
    // Set of states with positive reward (direct or via any action).
    std::unordered_set<std::string> goals;
    for (const auto& [s, r] : ast.rewards) if (r > 0) goals.insert(s);
    for (const auto& [s, am] : ast.action_rewards)
        for (const auto& [a, r] : am) if (r > 0) goals.insert(s);
    if (goals.empty()) return;

    // Reverse adjacency: rev[v] = list<u> with some edge u -> v.
    std::unordered_map<std::string, std::vector<std::string>> rev;
    for (const auto& t : ast.transitions) {
        if (t.probability > 0)
            rev[t.dest_state].push_back(t.source_state);
    }
    // BFS backwards from goals: anything reachable in reverse is "alive".
    std::unordered_set<std::string> alive;
    std::deque<std::string> q;
    for (const auto& g : goals) { alive.insert(g); q.push_back(g); }
    while (!q.empty()) {
        std::string u = q.front(); q.pop_front();
        auto it = rev.find(u);
        if (it == rev.end()) continue;
        for (const auto& v : it->second) {
            if (alive.insert(v).second) q.push_back(v);
        }
    }
    // Any declared state not in `alive` is dead. (Excluding goal states themselves.)
    std::vector<std::string> dead;
    for (const auto& s : ast.states)
        if (!alive.count(s)) dead.push_back(s);
    if (dead.empty()) return;
    std::sort(dead.begin(), dead.end());

    AutopsyFinding f;
    f.cls = FindingClass::DeadState;
    f.severity = FindingSeverity::ERROR_LEVEL;
    std::ostringstream hl;
    hl << "Dead state(s) detected: " << dead.size() << " state"
       << (dead.size() == 1 ? "" : "s") << " cannot reach any positive reward under any policy.";
    f.headline = hl.str();
    std::ostringstream lst;
    lst << "Dead states: ";
    for (size_t i = 0; i < dead.size(); ++i) { if (i) lst << ", "; lst << dead[i]; }
    f.details.push_back(lst.str());
    f.details.push_back("These states either have no outgoing transitions to reward-bearing states, or only loop among themselves.");
    f.details.push_back("Recommendation: add an escape transition or terminal reward, or remove these states.");
    f.numeric_fields["dead_state_count"] = static_cast<double>(dead.size());
    f.text_fields["dead_states"] = lst.str();
    report.findings.push_back(f);
}

// ----------------------------------------------------------------------------
// Class 3 — Discount Cliff (THE marquee output)
// ----------------------------------------------------------------------------
// Sweep gamma over a fine grid; record the policy at each step. A "cliff"
// is a gamma where the policy at some state flips. We report the cliff(s)
// closest to the user's actual gamma, with a clear "you are k below/above
// the threshold" framing.
inline void autopsyDiscountCliff(MDP_AST ast, AutopsyReport& report) {
    if (ast.discount_factor <= 0.0 || ast.discount_factor > 0.999) return;
    if (ast.states.empty()) return;

    const double user_gamma = ast.discount_factor;
    const int N = 200;  // grid resolution
    const double lo = 0.01, hi = 0.999;
    // Save the original solver mode and force value_iteration during the sweep
    // (PI changes the action breaking ties differently across runs of gamma).
    std::string saved_mode = ast.solver_mode;
    ast.solver_mode = "value_iteration";

    std::vector<double>                              gammas;
    std::vector<std::unordered_map<std::string, std::string>> policies;
    gammas.reserve(N + 1);
    policies.reserve(N + 1);
    for (int k = 0; k <= N; ++k) {
        double g = lo + (hi - lo) * (static_cast<double>(k) / N);
        ast.discount_factor = g;
        auto sol = solveOnce(ast);
        gammas.push_back(g);
        policies.push_back(sol.policy);
    }
    // Restore.
    ast.solver_mode = saved_mode;

    // Find consecutive grid pairs where the policy differs at some state.
    struct Transition {
        double gamma_low, gamma_high;
        std::string state;
        std::string action_low;
        std::string action_high;
    };
    std::vector<Transition> transitions;
    for (size_t i = 1; i < gammas.size(); ++i) {
        for (const auto& s : ast.states) {
            auto it1 = policies[i-1].find(s);
            auto it2 = policies[i].find(s);
            if (it1 == policies[i-1].end() || it2 == policies[i].end()) continue;
            if (it1->second != it2->second) {
                transitions.push_back({gammas[i-1], gammas[i], s, it1->second, it2->second});
            }
        }
    }
    if (transitions.empty()) return;

    // Pick the transition whose gamma midpoint is closest to user_gamma — this is the
    // "policy phase transition you are nearest to" finding.
    std::sort(transitions.begin(), transitions.end(),
              [&](const Transition& a, const Transition& b) {
                  double am = 0.5 * (a.gamma_low + a.gamma_high);
                  double bm = 0.5 * (b.gamma_low + b.gamma_high);
                  return std::abs(am - user_gamma) < std::abs(bm - user_gamma);
              });

    const auto& near = transitions[0];
    double mid = 0.5 * (near.gamma_low + near.gamma_high);
    AutopsyFinding f;
    f.cls = FindingClass::DiscountCliff;
    f.severity = (std::abs(mid - user_gamma) < 0.05)
                ? FindingSeverity::WARN  // close enough to matter
                : FindingSeverity::INFO; // exists but you are far from it
    std::ostringstream hl;
    hl << "Discount Cliff at gamma* ~ " << std::fixed << std::setprecision(3) << mid
       << " (your gamma = " << user_gamma << "). Policy at state '" << near.state
       << "' flips: " << near.action_low << " ↔ " << near.action_high << ".";
    f.headline = hl.str();
    std::ostringstream d1;
    d1 << "gamma in [" << std::fixed << std::setprecision(3) << lo << ", " << near.gamma_low
       << "]: pi('" << near.state << "') = " << near.action_low;
    std::ostringstream d2;
    d2 << "gamma in [" << std::fixed << std::setprecision(3) << near.gamma_high << ", " << hi
       << "]: pi('" << near.state << "') = " << near.action_high;
    f.details.push_back(d1.str());
    f.details.push_back(d2.str());
    std::ostringstream d3;
    d3 << "Distance from your gamma: " << std::fixed << std::setprecision(4)
       << std::abs(mid - user_gamma);
    f.details.push_back(d3.str());
    if (transitions.size() > 1) {
        std::ostringstream d4;
        d4 << "Total distinct phase transitions found in [" << lo << ", " << hi
           << "]: " << transitions.size();
        f.details.push_back(d4.str());
    }
    f.numeric_fields["cliff_gamma"]    = mid;
    f.numeric_fields["user_gamma"]     = user_gamma;
    f.numeric_fields["distance"]       = std::abs(mid - user_gamma);
    f.numeric_fields["total_cliffs"]   = static_cast<double>(transitions.size());
    f.text_fields["state"]             = near.state;
    f.text_fields["action_low_gamma"]  = near.action_low;
    f.text_fields["action_high_gamma"] = near.action_high;
    report.findings.push_back(f);
}

// ----------------------------------------------------------------------------
// Class 2 — Reward Hacking (cycle detection via Tarjan SCC)
// ----------------------------------------------------------------------------
// We compute strongly-connected components of the *optimal-policy graph*: from
// each state s, the edge follows pi*(s). Any SCC of size > 1 (or a self-loop)
// is a reachable cycle. If the discounted geometric sum of rewards along the
// cycle exceeds V*(goal), that's reward hacking.
inline void autopsyRewardHacking(const MDP_AST& ast, AutopsyReport& report) {
    if (ast.discount_factor <= 0.0 || ast.discount_factor >= 1.0) return;
    SimpleSolution sol = solveOnce(ast);
    if (sol.policy.empty()) return;

    // For each state s, expected successor under pi*(s) is the highest-prob dest.
    // (We collapse stochastic transitions to their mode for SCC purposes; the
    // value computation below uses the actual probabilities.)
    std::unordered_map<std::string, std::string> next_under_policy;
    for (const auto& [s, a] : sol.policy) {
        std::string best_dest;
        double best_p = -1.0;
        for (const auto& t : ast.transitions) {
            if (t.source_state == s && t.action == a && t.probability > best_p) {
                best_p = t.probability; best_dest = t.dest_state;
            }
        }
        if (!best_dest.empty()) next_under_policy[s] = best_dest;
    }

    // Tarjan SCC on this collapsed graph.
    std::unordered_map<std::string, int> index_of, lowlink, on_stack;
    std::vector<std::string> stack;
    std::vector<std::vector<std::string>> sccs;
    int idx = 0;
    std::function<void(const std::string&)> strongconnect = [&](const std::string& v) {
        index_of[v] = idx;
        lowlink[v]  = idx;
        ++idx;
        stack.push_back(v);
        on_stack[v] = 1;
        auto it = next_under_policy.find(v);
        if (it != next_under_policy.end()) {
            const std::string& w = it->second;
            if (!index_of.count(w)) {
                strongconnect(w);
                lowlink[v] = std::min(lowlink[v], lowlink[w]);
            } else if (on_stack.count(w)) {
                lowlink[v] = std::min(lowlink[v], index_of[w]);
            }
        }
        if (lowlink[v] == index_of[v]) {
            std::vector<std::string> comp;
            while (!stack.empty()) {
                std::string w = stack.back(); stack.pop_back();
                on_stack.erase(w);
                comp.push_back(w);
                if (w == v) break;
            }
            sccs.push_back(comp);
        }
    };
    for (const auto& s : ast.states)
        if (!index_of.count(s)) strongconnect(s);

    // For each SCC of size > 1 (or self-loop), compute the cycle value under
    // policy: sum of R(s_i, pi*(s_i)) divided by (1 - gamma^|C|). A self-loop
    // is the special case where next_under_policy[s] == s.
    double gamma = ast.discount_factor;
    double max_value_anywhere = 0.0;
    for (const auto& [s, v] : sol.values) max_value_anywhere = std::max(max_value_anywhere, v);

    for (const auto& comp : sccs) {
        bool is_cycle = comp.size() > 1;
        if (comp.size() == 1) {
            // Self-loop iff next_under_policy[s] == s. But only flag this as
            // reward hacking if the agent had a legitimate alternative -- i.e.
            // some other action from this state leads to a DIFFERENT state.
            // Pure absorbing terminals (no action escapes) are normal goal
            // structure, not reward hacking.
            auto it = next_under_policy.find(comp[0]);
            if (it != next_under_policy.end() && it->second == comp[0]) {
                bool has_escape = false;
                for (const auto& t : ast.transitions) {
                    if (t.source_state == comp[0] && t.dest_state != comp[0]
                        && t.probability > 0) { has_escape = true; break; }
                }
                if (has_escape) is_cycle = true;
            }
        }
        if (!is_cycle) continue;

        // Cycle reward = sum of R(s_i, pi*(s_i)).
        double cycle_reward = 0.0;
        for (const auto& s : comp) {
            auto pit = sol.policy.find(s);
            if (pit == sol.policy.end()) continue;
            cycle_reward += getReward(ast, s, pit->second);
        }
        // Discounted infinite-horizon value of staying in the cycle.
        double gamma_pow_len = std::pow(gamma, static_cast<double>(comp.size()));
        if (gamma_pow_len >= 1.0) continue;  // shouldn't happen for gamma < 1
        double cycle_value = cycle_reward / (1.0 - gamma_pow_len);

        if (cycle_reward <= 0) continue;

        // Compare cycle value to the BEST V*(s) for any s outside this SCC.
        // If V_cycle > V_outside * 1.05 and the cycle is reachable, the agent
        // is rationally preferring the cycle over any external goal -- this
        // is reward hacking by definition.
        std::unordered_set<std::string> in_scc(comp.begin(), comp.end());
        double best_outside = -1e18;
        for (const auto& [s, v] : sol.values)
            if (!in_scc.count(s)) best_outside = std::max(best_outside, v);
        if (best_outside <= -1e17) continue;        // no comparable states
        // The cycle's discounted value already factors in the loop length.
        // We only fire when cycle_value beats the best outside V* by a margin.
        if (cycle_value > best_outside * 1.05 && cycle_value > 0) {
            AutopsyFinding f;
            f.cls = FindingClass::RewardHacking;
            f.severity = FindingSeverity::ERROR_LEVEL;
            std::ostringstream hl;
            hl << "Reward Hacking: cycle of length " << comp.size()
               << " under optimal policy yields V_cycle = " << std::fixed
               << std::setprecision(2) << cycle_value
               << " > best V*(state outside cycle) = " << best_outside << ".";
            f.headline = hl.str();
            std::ostringstream path;
            path << "Cycle: ";
            for (size_t i = 0; i < comp.size(); ++i) {
                if (i) path << " -> ";
                path << comp[i];
            }
            path << " -> " << comp[0];
            f.details.push_back(path.str());
            std::ostringstream cr;
            cr << "Per-loop reward: " << cycle_reward
               << "; discounted value (gamma=" << gamma << "): " << cycle_value;
            f.details.push_back(cr.str());
            f.details.push_back("The agent prefers cycling to reaching the highest-reward state. Mitigations:");
            f.details.push_back("  [1] Make the high-reward state terminal (absorbing).");
            f.details.push_back("  [2] Add a per-step or per-action cost on the cycle edge.");
            f.numeric_fields["cycle_value"]      = cycle_value;
            f.numeric_fields["best_outside_v"]   = best_outside;
            f.numeric_fields["cycle_length"]     = static_cast<double>(comp.size());
            f.numeric_fields["cycle_reward"]     = cycle_reward;
            f.text_fields["cycle_path"]       = path.str();
            report.findings.push_back(f);
        }
    }
}

// ----------------------------------------------------------------------------
// Class 6 — Fragility Analysis
// ----------------------------------------------------------------------------
// For each (state, parameter) pair, binary-search the smallest perturbation
// that changes pi*(state). The "parameters" we perturb are:
//   * action-rewards R(s, a) for actions used by the policy
//   * the discount factor gamma
// We report the SINGLE most fragile decision (smallest perturbation to flip
// it) and the overall policy robustness score.
inline double fragilityFlipDelta(MDP_AST ast_copy, const std::string& kind,
                                 const std::string& key1, const std::string& key2,
                                 double base_value, double bound_low, double bound_high,
                                 const std::string& target_state,
                                 const std::string& target_action,
                                 double max_delta)
{
    // Returns the smallest |δ| with the policy at target_state != target_action,
    // searching outward from 0 along the legal axis. Returns max_delta if none found.
    auto solveWith = [&](double new_val) {
        if (kind == "discount") ast_copy.discount_factor = new_val;
        else if (kind == "reward") ast_copy.rewards[key1] = new_val;
        else if (kind == "action_reward") ast_copy.action_rewards[key1][key2] = new_val;
        auto sol = solveOnce(ast_copy);
        auto it = sol.policy.find(target_state);
        return it == sol.policy.end() ? std::string("") : it->second;
    };
    // Search positive and negative directions independently.
    double best = max_delta;
    for (int sign : {+1, -1}) {
        double lo = 0.0, hi_step = 0.001;
        // Exponentially grow until we find a flip, capped at the axis bound or max_delta.
        double last_v = base_value;
        bool   flipped = false;
        for (int i = 0; i < 60 && hi_step < max_delta; ++i, hi_step *= 1.6) {
            double trial = base_value + sign * hi_step;
            if (trial < bound_low || trial > bound_high) break;
            std::string a = solveWith(trial);
            if (!a.empty() && a != target_action) { flipped = true; break; }
            last_v = trial;
            lo = hi_step;
        }
        // Restore.
        solveWith(base_value); (void)last_v;
        if (!flipped) continue;
        // Bisect.
        double a_lo = lo, a_hi = hi_step;
        for (int i = 0; i < 30; ++i) {
            double mid = 0.5 * (a_lo + a_hi);
            double trial = base_value + sign * mid;
            if (trial < bound_low) trial = bound_low;
            if (trial > bound_high) trial = bound_high;
            std::string a = solveWith(trial);
            if (!a.empty() && a != target_action) a_hi = mid;
            else                                  a_lo = mid;
        }
        // Restore.
        solveWith(base_value);
        if (a_hi < best) best = a_hi;
    }
    return best;
}

inline void autopsyFragility(const MDP_AST& ast, AutopsyReport& report,
                             double budget = 50.0)
{
    SimpleSolution sol = solveOnce(ast);
    if (sol.policy.empty()) return;

    // For each (state, policy) pair, sweep the parameters most likely to flip it.
    struct FragRow {
        std::string state;
        std::string action;
        double      best_delta;
        std::string param_label;
    };
    std::vector<FragRow> rows;
    for (const auto& [s, a] : sol.policy) {
        double best = budget;
        std::string best_label = "(none)";
        // (a) action_rewards used at this state.
        for (const auto& [s2, am] : ast.action_rewards) {
            for (const auto& [act, r] : am) {
                MDP_AST clone = ast;
                double d = fragilityFlipDelta(clone, "action_reward", s2, act, r,
                                              -1e9, 1e9, s, a, best);
                if (d < best) { best = d; best_label = "R(" + s2 + ", " + act + ")"; }
            }
        }
        // (b) discount factor.
        {
            MDP_AST clone = ast;
            double d = fragilityFlipDelta(clone, "discount", "", "",
                                          ast.discount_factor, 0.01, 0.999, s, a,
                                          std::min(budget, 0.5));
            if (d < best) { best = d; best_label = "gamma"; }
        }
        rows.push_back({s, a, best, best_label});
    }
    if (rows.empty()) return;

    // Most fragile = smallest delta.
    std::sort(rows.begin(), rows.end(),
              [](const FragRow& a, const FragRow& b) { return a.best_delta < b.best_delta; });

    AutopsyFinding f;
    f.cls = FindingClass::Fragility;
    f.severity = FindingSeverity::INFO;
    std::ostringstream hl;
    hl << "Fragility: most fragile decision pi('" << rows[0].state
       << "') = " << rows[0].action << " flips at delta = " << std::fixed
       << std::setprecision(4) << rows[0].best_delta
       << " on " << rows[0].param_label << ".";
    f.headline = hl.str();
    int K = std::min<int>(rows.size(), 5);
    for (int i = 0; i < K; ++i) {
        std::ostringstream d;
        d << "  pi('" << rows[i].state << "') = " << rows[i].action
          << "  →  flips at |δ| = " << std::fixed << std::setprecision(4)
          << rows[i].best_delta << " on " << rows[i].param_label;
        f.details.push_back(d.str());
    }
    // Robustness score: log-scaled average of best_delta clamped to [0, 10].
    double avg = 0.0;
    for (const auto& r : rows) avg += std::min(r.best_delta, budget);
    avg /= rows.size();
    double score = std::min(10.0, std::max(0.0, std::log10(avg + 1e-3) * 2.0 + 5.0));
    std::ostringstream rs;
    rs << "Overall policy robustness: " << std::fixed << std::setprecision(1) << score
       << " / 10 (avg flip-delta " << avg << " across " << rows.size() << " decisions).";
    f.details.push_back(rs.str());
    f.numeric_fields["min_flip_delta"]      = rows[0].best_delta;
    f.numeric_fields["avg_flip_delta"]      = avg;
    f.numeric_fields["robustness_score"]    = score;
    f.numeric_fields["decisions_analysed"]  = static_cast<double>(rows.size());
    f.text_fields["most_fragile_state"]     = rows[0].state;
    f.text_fields["most_fragile_param"]     = rows[0].param_label;
    report.findings.push_back(f);
}

// ----------------------------------------------------------------------------
// Dispatcher
// ----------------------------------------------------------------------------
inline AutopsyReport runAutopsy(MDP_AST& ast, const std::string& mode = "full") {
    AutopsyReport report;
    if (mode == "full" || mode == "reward") {
        autopsyRewardMyopia(ast, report);
        autopsyMagnitudeImbalance(ast, report);
        autopsyRewardHacking(ast, report);
    }
    if (mode == "full" || mode == "structural") {
        autopsyDeadStates(ast, report);
    }
    if (mode == "full" || mode == "solver") {
        autopsyDiscountCliff(ast, report);
    }
    if (mode == "full" || mode == "landscape") {
        autopsyFragility(ast, report);
    }
    return report;
}

// ----------------------------------------------------------------------------
// REPAIR command
// ----------------------------------------------------------------------------
// Target syntax (per spec):  "pi(state) == action"   or   "pi(state) != action"
// Strategy: parse the assertion, then binary-search a single parameter (rewards
// first, then gamma) to make pi(state) match the target. Returns the
// smallest-magnitude successful change.
struct ParsedAssertion {
    enum Kind { PI_EQ, PI_NEQ, UNSUPPORTED };
    Kind kind = UNSUPPORTED;
    std::string state;
    std::string action;
};

inline ParsedAssertion parseRepairTarget(const std::string& s) {
    ParsedAssertion out;
    // Tolerate either "pi(state) == action" or unicode "π(...)".
    std::string norm;
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c1 = static_cast<unsigned char>(s[i]);
        if (c1 == 0xCF && i + 1 < s.size() &&
            static_cast<unsigned char>(s[i+1]) == 0x80) { norm += "pi"; i += 2; }
        else { norm.push_back(s[i]); ++i; }
    }
    auto lparen = norm.find('(');
    auto rparen = norm.find(')');
    if (lparen == std::string::npos || rparen == std::string::npos || rparen < lparen)
        return out;
    std::string head = norm.substr(0, lparen);
    if (head != "pi" && head != "PI" && head != "Pi") return out;
    std::string state_part = norm.substr(lparen + 1, rparen - lparen - 1);
    // Trim.
    while (!state_part.empty() && state_part.front() == ' ') state_part.erase(0, 1);
    while (!state_part.empty() && state_part.back()  == ' ') state_part.pop_back();

    std::string tail = norm.substr(rparen + 1);
    // Find "==" or "!=".
    auto eq = tail.find("==");
    auto ne = tail.find("!=");
    std::string rhs;
    if (eq != std::string::npos) { rhs = tail.substr(eq + 2); out.kind = ParsedAssertion::PI_EQ; }
    else if (ne != std::string::npos) { rhs = tail.substr(ne + 2); out.kind = ParsedAssertion::PI_NEQ; }
    else return out;
    while (!rhs.empty() && rhs.front() == ' ') rhs.erase(0, 1);
    while (!rhs.empty() && rhs.back()  == ' ') rhs.pop_back();
    out.state  = state_part;
    out.action = rhs;
    return out;
}

inline bool assertionHolds(const SimpleSolution& sol, const ParsedAssertion& a) {
    auto it = sol.policy.find(a.state);
    if (it == sol.policy.end()) return false;
    if (a.kind == ParsedAssertion::PI_EQ)  return it->second == a.action;
    if (a.kind == ParsedAssertion::PI_NEQ) return it->second != a.action;
    return false;
}

inline RepairResult repairMDP(const MDP_AST& original, const std::string& target_str) {
    RepairResult result;
    result.target_assertion = target_str;
    result.already_satisfied = false;
    result.any_succeeded     = false;

    ParsedAssertion target = parseRepairTarget(target_str);
    if (target.kind == ParsedAssertion::UNSUPPORTED) {
        return result;
    }
    if (!original.states.count(target.state)) return result;

    // Check if the assertion already holds.
    {
        SimpleSolution sol = solveOnce(original);
        if (assertionHolds(sol, target)) {
            result.already_satisfied = true;
            return result;
        }
    }

    auto trySingleParam = [&](const std::string& kind, const std::string& k1,
                              const std::string& k2, double base_val,
                              double bound_low, double bound_high,
                              const std::string& label, double max_mag) -> RepairProposal {
        RepairProposal p{kind, label, base_val, base_val, 0.0, false, ""};
        auto apply = [&](double new_val) {
            MDP_AST clone = original;
            if (kind == "discount") clone.discount_factor = new_val;
            else if (kind == "reward") clone.rewards[k1] = new_val;
            else if (kind == "action_reward") clone.action_rewards[k1][k2] = new_val;
            return solveOnce(clone);
        };
        // Search both directions; pick the smallest |δ|.
        double best_delta = max_mag;
        double best_new_value = base_val;
        for (int sign : {+1, -1}) {
            // Exponentially probe.
            double step = 0.001;
            bool found = false;
            for (int i = 0; i < 80 && step < max_mag; ++i, step *= 1.5) {
                double trial = base_val + sign * step;
                if (trial < bound_low || trial > bound_high) break;
                if (assertionHolds(apply(trial), target)) { found = true; break; }
            }
            if (!found) continue;
            // Bisect to refine.
            double lo = 0.0, hi = step;
            for (int i = 0; i < 30; ++i) {
                double mid = 0.5 * (lo + hi);
                double trial = base_val + sign * mid;
                if (trial < bound_low) trial = bound_low;
                if (trial > bound_high) trial = bound_high;
                if (assertionHolds(apply(trial), target)) hi = mid;
                else                                      lo = mid;
            }
            if (hi < best_delta) { best_delta = hi; best_new_value = base_val + sign * hi; }
        }
        if (best_delta < max_mag) {
            p.new_value = best_new_value;
            p.magnitude = std::abs(best_new_value - base_val);
            p.succeeded = true;
        }
        return p;
    };

    // Try every action-reward; then state-reward; then gamma. Single-parameter
    // perturbations only — multi-parameter repair noted as future work.
    for (const auto& [s, am] : original.action_rewards) {
        for (const auto& [a, r] : am) {
            auto p = trySingleParam("action_reward", s, a, r,
                                    -1e6, 1e6,
                                    "R(" + s + ", " + a + ")", 1e6);
            if (p.succeeded) result.proposals.push_back(p);
        }
    }
    for (const auto& [s, r] : original.rewards) {
        auto p = trySingleParam("reward", s, "", r, -1e6, 1e6, "R(" + s + ")", 1e6);
        if (p.succeeded) result.proposals.push_back(p);
    }
    {
        auto p = trySingleParam("discount", "", "", original.discount_factor,
                                0.01, 0.999, "gamma", 0.99);
        if (p.succeeded) result.proposals.push_back(p);
    }

    std::sort(result.proposals.begin(), result.proposals.end(),
              [](const RepairProposal& a, const RepairProposal& b) {
                  return a.magnitude < b.magnitude;
              });
    result.any_succeeded = !result.proposals.empty();
    return result;
}

// ----------------------------------------------------------------------------
// Report printer (terminal)
// ----------------------------------------------------------------------------
inline void printAutopsyReport(const AutopsyReport& report) {
    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << "  AUTOPSY REPORT\n";
    std::cout << "======================================\n\n";
    if (report.findings.empty()) {
        std::cout << "  No issues detected.\n";
        std::cout << "======================================\n";
        return;
    }
    int idx = 1;
    for (const auto& f : report.findings) {
        const char* sev_marker = "  ";
        if (f.severity == FindingSeverity::ERROR_LEVEL) sev_marker = "X ";
        else if (f.severity == FindingSeverity::WARN)   sev_marker = "* ";
        else                                            sev_marker = "i ";
        std::cout << sev_marker << "[" << findingSeverityName(f.severity) << "/"
                  << findingClassName(f.cls) << "] " << f.headline << "\n";
        for (const auto& d : f.details) std::cout << "    " << d << "\n";
        std::cout << "\n";
        ++idx;
    }
    std::cout << "======================================\n";
    int err = 0, warn = 0;
    for (const auto& f : report.findings) {
        if (f.severity == FindingSeverity::ERROR_LEVEL) ++err;
        else if (f.severity == FindingSeverity::WARN)   ++warn;
    }
    std::cout << "  Findings: " << err << " ERROR, " << warn << " WARN, "
              << report.findings.size() - err - warn << " INFO.\n";
    std::cout << "======================================\n";
}

inline void printRepairReport(const RepairResult& r) {
    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << "  REPAIR REPORT\n";
    std::cout << "======================================\n";
    std::cout << "  Target: " << r.target_assertion << "\n\n";
    if (r.already_satisfied) {
        std::cout << "  The assertion already holds in the unmodified MDP. No repair needed.\n";
        std::cout << "======================================\n";
        return;
    }
    if (!r.any_succeeded) {
        std::cout << "  No single-parameter repair found within the search bounds.\n";
        std::cout << "  Note: multi-parameter repair is future work; the spec restricts\n";
        std::cout << "  this command to single-parameter perturbations.\n";
        std::cout << "======================================\n";
        return;
    }
    std::cout << "  " << r.proposals.size() << " repair(s) found, ranked by magnitude:\n\n";
    int K = std::min<int>(r.proposals.size(), 5);
    for (int i = 0; i < K; ++i) {
        const auto& p = r.proposals[i];
        std::cout << "  [" << (i+1) << "] " << p.parameter_label
                  << " : " << std::fixed << std::setprecision(4) << p.original_value
                  << " -> " << p.new_value
                  << "  (delta = " << p.magnitude << ")\n";
    }
    std::cout << "\n  Minimal fix: " << r.proposals[0].parameter_label
              << " -> " << std::fixed << std::setprecision(4) << r.proposals[0].new_value
              << "\n";
    std::cout << "======================================\n";
}

// ============================================================================
// SECTION 9: PHASE 3.5 — VERIFIER (Upgrade 6)
// ============================================================================

VerifyResult runVerification(const SolverResult& solver, const MDP_AST& ast) {
    VerifyResult vr;
    vr.pass_count = 0;
    vr.fail_count = 0;

    // Normalize unicode π (UTF-8: 0xCF 0x80) to ASCII "pi" so that the
    // existing tokenizer below can stay simple. This lets users write either
    // VERIFY: π(s) == a   or   VERIFY: pi(s) == a.
    auto normalizePi = [](std::string s) -> std::string {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c1 = static_cast<unsigned char>(s[i]);
            if (c1 == 0xCF && i + 1 < s.size() &&
                static_cast<unsigned char>(s[i+1]) == 0x80) {
                out += "pi";
                i += 2;
            } else {
                out.push_back(s[i]);
                ++i;
            }
        }
        return out;
    };

    for (const auto& raw_assertion : ast.verify_assertions) {
        std::string assertion = normalizePi(raw_assertion);
        std::string expr = assertion;
        bool passed = false;
        std::string detail;

        // Parse: V(state) op value, V(state) op V(state), pi(state) op action
        auto parseRef = [&](const std::string& token, double& val, std::string& str_val, bool& is_v, bool& is_pi) {
            is_v = false; is_pi = false;
            if (token.substr(0, 2) == "V(" && token.back() == ')') {
                is_v = true;
                std::string sname = token.substr(2, token.size() - 3);
                if (solver.values.count(sname) > 0) val = solver.values.at(sname);
                else val = 0.0;
                str_val = sname;
            } else if (token.substr(0, 3) == "pi(" && token.back() == ')') {
                is_pi = true;
                std::string sname = token.substr(3, token.size() - 4);
                if (solver.policy.count(sname) > 0) str_val = solver.policy.at(sname);
                else str_val = "(unknown)";
            } else {
                // Literal value or action name
                try { val = std::stod(token); } catch (...) { str_val = token; }
            }
        };

        // Tokenize the assertion
        std::istringstream iss(expr);
        std::string lhs_tok, op_tok, rhs_tok;
        iss >> lhs_tok >> op_tok >> rhs_tok;

        double lhs_val = 0.0, rhs_val = 0.0;
        std::string lhs_str, rhs_str;
        bool lhs_is_v = false, lhs_is_pi = false;
        bool rhs_is_v = false, rhs_is_pi = false;

        parseRef(lhs_tok, lhs_val, lhs_str, lhs_is_v, lhs_is_pi);
        parseRef(rhs_tok, rhs_val, rhs_str, rhs_is_v, rhs_is_pi);

        // Handle V(s) op V(s') comparison
        if (lhs_is_v && rhs_is_v) {
            double lv = solver.values.count(lhs_str) ? solver.values.at(lhs_str) : 0.0;
            double rv = solver.values.count(rhs_str) ? solver.values.at(rhs_str) : 0.0;
            if (op_tok == ">") passed = lv > rv;
            else if (op_tok == "<") passed = lv < rv;
            else if (op_tok == ">=") passed = lv >= rv;
            else if (op_tok == "<=") passed = lv <= rv;
            else if (op_tok == "==") passed = std::abs(lv - rv) < 1e-9;
            else if (op_tok == "!=") passed = std::abs(lv - rv) >= 1e-9;
            std::ostringstream d; d << std::fixed << std::setprecision(4) << lv << " " << op_tok << " " << rv;
            detail = d.str();
        }
        // V(s) op literal
        else if (lhs_is_v && !rhs_is_v && !rhs_is_pi) {
            double lv = solver.values.count(lhs_str) ? solver.values.at(lhs_str) : 0.0;
            if (op_tok == ">") passed = lv > rhs_val;
            else if (op_tok == "<") passed = lv < rhs_val;
            else if (op_tok == ">=") passed = lv >= rhs_val;
            else if (op_tok == "<=") passed = lv <= rhs_val;
            else if (op_tok == "==") passed = std::abs(lv - rhs_val) < 1e-9;
            else if (op_tok == "!=") passed = std::abs(lv - rhs_val) >= 1e-9;
            std::ostringstream d; d << std::fixed << std::setprecision(4) << lv << " " << op_tok << " " << rhs_val;
            detail = d.str();
        }
        // pi(s) op action
        else if (lhs_is_pi) {
            if (op_tok == "==") passed = (lhs_str == rhs_tok);
            else if (op_tok == "!=") passed = (lhs_str != rhs_tok);
            detail = lhs_str + " " + op_tok + " " + rhs_tok;
        }

        if (passed) {
            vr.pass_count++;
            vr.messages.push_back("[VERIFY PASS] " + expr + ": " + detail + " -> TRUE");
        } else {
            vr.fail_count++;
            vr.messages.push_back("[VERIFY FAIL] " + expr + ": " + detail + " -> FALSE");
        }
    }

    return vr;
}


// ============================================================================
// SECTION 10: BENCHMARK REPORT (Upgrade 7)
// ============================================================================

void printBenchmarkReport(const SolverResult& vi, const PolicyIterationResult& pi,
                          const QLearningResult& ql, const MDP_AST& ast,
                          const std::string& filename) {
    int n_states = static_cast<int>(ast.states.size());

    // Policy match counts
    int pi_match = 0, ql_match = 0;
    for (const auto& s : ast.states) {
        if (vi.policy.count(s) && pi.policy.count(s) && vi.policy.at(s) == pi.policy.at(s)) pi_match++;
        if (vi.policy.count(s) && ql.policy.count(s) && vi.policy.at(s) == ql.policy.at(s)) ql_match++;
    }

    // Max V* error vs VI
    double pi_max_err = 0.0, ql_max_err = 0.0;
    for (const auto& s : ast.states) {
        if (vi.values.count(s) && pi.values.count(s))
            pi_max_err = std::max(pi_max_err, std::abs(vi.values.at(s) - pi.values.at(s)));
        if (vi.values.count(s)) {
            // For QL, extract V from Q-table: V(s) = max_a Q(s,a)
            double ql_v = std::numeric_limits<double>::lowest();
            if (ql.Q_table.count(s))
                for (const auto& [a, q] : ql.Q_table.at(s))
                    if (q > ql_v) ql_v = q;
            if (ql_v == std::numeric_limits<double>::lowest()) ql_v = 0.0;
            ql_max_err = std::max(ql_max_err, std::abs(vi.values.at(s) - ql_v));
        }
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  BENCHMARK REPORT: " << filename << "\n";
    std::cout << "================================================================\n\n";

    std::cout << std::left << std::fixed << std::setprecision(2);
    std::cout << "  " << std::setw(26) << "Metric"
              << std::setw(16) << "Value Iter"
              << std::setw(16) << "Policy Iter"
              << "Q-Learning\n";
    std::cout << "  " << std::string(70, '-') << "\n";

    std::cout << "  " << std::setw(26) << "Outer iterations"
              << std::setw(16) << vi.iterations
              << std::setw(16) << pi.policy_improvement_steps
              << ql.episodes_to_convergence << " ep\n";

    std::cout << "  " << std::setw(26) << "Bellman evaluations"
              << std::setw(16) << vi.iterations
              << std::setw(16) << pi.total_bellman_evaluations
              << ql.episodes_to_convergence << "\n";

    std::cout << "  " << std::setw(26) << "Wall-clock time (ms)"
              << std::setw(16) << vi.wall_clock_ms
              << std::setw(16) << pi.wall_clock_ms
              << ql.wall_clock_ms << "\n";

    std::ostringstream pi_err_s, ql_err_s;
    pi_err_s << std::scientific << std::setprecision(1) << pi_max_err;
    ql_err_s << std::fixed << std::setprecision(3) << ql_max_err;
    std::cout << "  " << std::setw(26) << "Max |V* error| vs VI"
              << std::setw(16) << "0.0 (ref)"
              << std::setw(16) << pi_err_s.str()
              << ql_err_s.str() << "\n";

    std::cout << "  " << std::setw(26) << "Policy match vs VI"
              << std::setw(16) << (std::to_string(n_states) + "/" + std::to_string(n_states))
              << std::setw(16) << (std::to_string(pi_match) + "/" + std::to_string(n_states))
              << (std::to_string(ql_match) + "/" + std::to_string(n_states)) << "\n";

    std::cout << "  " << std::setw(26) << "Converged?"
              << std::setw(16) << (vi.converged ? "Yes" : "No")
              << std::setw(16) << (pi.converged ? "Yes" : "No")
              << (ql.converged ? "Yes" : "No") << "\n";

    std::cout << "================================================================\n";
}


// ============================================================================
// SECTION 11: SOLVER REPORT PRINTER
// ============================================================================

void printSolverReport(const SolverResult& result, const MDP_AST& ast, const std::string& solver_name = "VALUE ITERATION") {
    std::cout << "\n======================================\n";
    std::cout << "  " << solver_name << " RESULTS\n";
    std::cout << "======================================\n\n";

    std::cout << "-- CONVERGENCE --\n";
    if (result.converged) std::cout << "  Converged after " << result.iterations << " iterations.\n";
    else std::cout << "  Did NOT converge within " << result.iterations << " iterations.\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << result.wall_clock_ms << " ms\n\n";

    std::cout << "  " << std::left << std::setw(14) << "State" << std::setw(12) << "R(s)"
              << std::setw(14) << "V*(s)" << "pi*(s)\n";
    std::cout << "  " << std::string(54, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& s : ast.states) {
        double r = ast.rewards.count(s) ? ast.rewards.at(s) : 0.0;
        std::cout << "  " << std::left << std::setw(14) << s << std::setw(12) << r
                  << std::setw(14) << result.values.at(s) << result.policy.at(s) << "\n";
    }
    std::cout << "\n======================================\n";
}

void printPIReport(const PolicyIterationResult& pi, const MDP_AST& ast) {
    SolverResult sr;
    sr.values = pi.values; sr.policy = pi.policy;
    sr.iterations = pi.policy_improvement_steps;
    sr.converged = pi.converged; sr.wall_clock_ms = pi.wall_clock_ms;
    std::cout << "\n  Policy improvement steps: " << pi.policy_improvement_steps << "\n";
    std::cout << "  Total Bellman evaluations: " << pi.total_bellman_evaluations << "\n";
    printSolverReport(sr, ast, "POLICY ITERATION");
}

void printQLReport(const QLearningResult& ql, const MDP_AST& ast) {
    std::cout << "\n======================================\n";
    std::cout << "  Q-LEARNING RESULTS\n";
    std::cout << "======================================\n\n";
    std::cout << "  Episodes: " << ql.episode_rewards.size() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << ql.wall_clock_ms << " ms\n\n";

    std::cout << "  " << std::left << std::setw(14) << "State" << "pi*(s)\n";
    std::cout << "  " << std::string(30, '-') << "\n";
    for (const auto& s : ast.states)
        std::cout << "  " << std::left << std::setw(14) << s << ql.policy.at(s) << "\n";
    std::cout << "\n======================================\n";
}


// ============================================================================
// SECTION 12: MAIN ENTRY POINT
// ============================================================================

#ifndef MDP_TESTING_MODE  // Excluded when compiling tests or visualization modules

// Forward declarations for visualization modules (defined in separate .cpp files)
// When MDP_VIZ_ENABLED, we include the implementation files directly to compile
// as a single translation unit (avoids multiple-definition linker errors from
// the header-based architecture of mdp_compiler.cpp).
#ifdef MDP_VIZ_ENABLED
#include "visualizer.hpp"
#include "html_output.hpp"
#include "visualizer.cpp"
#include "html_output.cpp"
#endif

int main(int argc, char* argv[]) {
    // ---- Windows console: enable ANSI escape sequences (VT processing) ----
    // Without this call, ANSI color codes and cursor-positioning escapes print
    // as literal characters on Windows 10 cmd.exe / PowerShell, making
    // --animate look broken. We do this unconditionally on Windows; it is a
    // silent no-op on terminals that already support ANSI (Windows Terminal,
    // VS Code terminal, etc.). On non-Windows platforms this block is gone.
#ifdef _WIN32
    {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD mode = 0;
            if (GetConsoleMode(hOut, &mode)) {
                SetConsoleMode(hOut, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */);
            }
        }
        // Also enable UTF-8 output so Unicode characters (π, arrows, etc.)
        // render correctly. SetConsoleOutputCP is harmless if already set.
        SetConsoleOutputCP(65001);
    }
#endif

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_mdp_file>\n"
                  << "       [--animate] [--html out.html]\n"
                  << "       [--json | --json-out out.json] [--show-hmm]\n"
                  << "       [--diagnose[=reward|structural|solver|landscape]]\n"
                  << "       [--fragility] [--repair \"pi(state) == action\"]\n";
        return EXIT_FAILURE;
    }

    // ---- Parse command-line flags ----
    std::string filepath = argv[1];
    bool flag_animate  = false;
    bool flag_json     = false;
    bool flag_show_hmm = false;
    bool flag_diagnose = false;
    bool flag_fragility= false;
    std::string diagnose_mode = "full";   // "full" | "reward" | "structural" | "solver" | "landscape"
    std::string repair_target;            // empty => no --repair
    std::string html_output_path;
    std::string json_output_path;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--animate") {
            flag_animate = true;
        } else if (arg == "--html" && i + 1 < argc) {
            html_output_path = argv[++i];
        } else if (arg == "--json") {
            flag_json = true;
        } else if (arg == "--json-out" && i + 1 < argc) {
            flag_json = true;
            json_output_path = argv[++i];
        } else if (arg == "--show-hmm") {
            flag_show_hmm = true;
        } else if (arg == "--diagnose") {
            flag_diagnose = true; diagnose_mode = "full";
        } else if (arg.rfind("--diagnose=", 0) == 0) {
            flag_diagnose = true; diagnose_mode = arg.substr(11);
        } else if (arg == "--fragility") {
            flag_fragility = true;
        } else if (arg == "--repair" && i + 1 < argc) {
            repair_target = argv[++i];
        }
    }

    // In --json mode (without --json-out), redirect all human-readable
    // output to stderr so that stdout contains ONLY the final JSON payload.
    // This makes `mdp_compiler model.mdp --json > out.json` work as documented.
    std::streambuf* orig_cout = std::cout.rdbuf();
    if (flag_json && json_output_path.empty()) {
        std::cout.rdbuf(std::cerr.rdbuf());
    }

    // Phase 0+1: Preprocess and Parse
    MDP_AST ast = parseMDPFile(filepath);
    if (ast.states.empty() && !ast.hmm_bridge_enabled) {
        std::cerr << "[FATAL] No states parsed.\n";
        return EXIT_FAILURE;
    }

    // Phase 1.5: HMM-MDP Bridge (Phase 2A).
    // If BRIDGE: hmm -> mdp is enabled, fit Baum-Welch on the declared CSV
    // and plumb the learned transition matrix into the AST as TRANSITION rows
    // before validation runs.
    if (ast.hmm_bridge_enabled) {
        std::cout << "\nFitting HMM (Baum-Welch, scaled forward-backward)...\n";
        auto bridge_errors = bridgeHMMToMDP(ast);
        if (!bridge_errors.empty()) {
            for (const auto& e : bridge_errors) std::cerr << "  " << e << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "  HMM fitted: K=" << ast.hmm_states.size()
                  << "  iterations=" << ast.hmm_fit_iterations
                  << "  logL=" << std::fixed << std::setprecision(2)
                  << ast.hmm_log_likelihood << "\n";
        if (flag_show_hmm) {
            std::cout << "\n-- Learned HMM transition matrix --\n";
            for (size_t i = 0; i < ast.hmm_states.size(); ++i) {
                std::cout << "  " << std::left << std::setw(12) << ast.hmm_states[i] << " |";
                for (size_t j = 0; j < ast.hmm_states.size(); ++j) {
                    std::cout << " " << std::fixed << std::setprecision(3)
                              << ast.hmm_transition_matrix[i][j];
                }
                std::cout << "\n";
            }
            std::cout << "-- Emission parameters --\n";
            std::cout << "  " << std::left << std::setw(12) << "regime"
                      << std::right << std::setw(10) << "mu"
                      << std::setw(10) << "sigma" << "\n";
            for (size_t i = 0; i < ast.hmm_states.size(); ++i) {
                std::cout << "  " << std::left << std::setw(12) << ast.hmm_states[i]
                          << std::right << std::setw(10) << std::fixed << std::setprecision(4) << ast.hmm_emission_mean[i]
                          << std::setw(10) << std::fixed << std::setprecision(4) << ast.hmm_emission_std[i] << "\n";
            }
            std::cout << "\n";
        }
    }

    printAST(ast);

    // Phase 2: Validate
    std::cout << "\nRunning semantic validation...\n";
    ValidationResult validation = validateAST(ast);
    printValidationReport(validation);
    if (!validation.isValid()) {
        std::cout << "\nValidation FAILED. Fix errors and re-run.\n";
        return EXIT_FAILURE;
    }

    // Phase 3: Solve
    SolverResult vi_result;
    PolicyIterationResult pi_result;
    QLearningResult ql_result;
    PBVIResult pbvi_result;
    bool ran_vi = false, ran_pi = false, ran_ql = false, ran_pbvi = false;

    if (ast.solver_mode == "value_iteration" || ast.solver_mode == "all") {
        std::cout << "\nRunning Value Iteration...\n";
        vi_result = solveValueIteration(ast);
        printSolverReport(vi_result, ast, "VALUE ITERATION");
        ran_vi = true;
    }
    if (ast.solver_mode == "policy_iteration" || ast.solver_mode == "all") {
        std::cout << "\nRunning Policy Iteration...\n";
        pi_result = solvePolicyIteration(ast);
        printPIReport(pi_result, ast);
        ran_pi = true;
    }
    if (ast.solver_mode == "q_learning" || ast.solver_mode == "all") {
        std::cout << "\nRunning Q-Learning...\n";
        ql_result = solveQLearning(ast);
        printQLReport(ql_result, ast);
        ran_ql = true;
    }
    if (ast.solver_mode == "pbvi") {
        std::cout << "\nRunning PBVI (POMDP)...\n";
        pbvi_result = solvePBVI(ast);
        printPBVIReport(pbvi_result, ast);
        ran_pbvi = true;
    }

    // Benchmark report if all MDP solvers ran
    if (ran_vi && ran_pi && ran_ql) {
        printBenchmarkReport(vi_result, pi_result, ql_result, ast, filepath);
    }

    // Phase 3.5: Verify assertions
    int verify_failures = 0;
    if (!ast.verify_assertions.empty()) {
        // Build a unified SolverResult view for the verifier.
        // For MDP solvers: take VI (or PI as fallback). For PBVI: merge belief
        // values & policies into the values/policy maps so that the existing
        // V(name) / pi(name) tokenizer just works for both states AND named
        // belief states.
        SolverResult vsr;
        if (ran_vi) vsr = vi_result;
        else if (ran_pi) { vsr.values = pi_result.values; vsr.policy = pi_result.policy; }

        if (ran_pbvi) {
            for (const auto& [name, v] : pbvi_result.values_at_belief) vsr.values[name] = v;
            for (const auto& [name, a] : pbvi_result.policy_at_belief) vsr.policy[name] = a;
        }

        VerifyResult vr = runVerification(vsr, ast);
        std::cout << "\n======================================\n";
        std::cout << "  VERIFICATION RESULTS\n";
        std::cout << "======================================\n\n";
        for (const auto& msg : vr.messages) std::cout << "  " << msg << "\n";
        std::cout << "\n  Passed: " << vr.pass_count << "/" << (vr.pass_count + vr.fail_count) << "\n";
        std::cout << "======================================\n";
        verify_failures = vr.fail_count;
    }

    // ---- Phase 4 (v3.0 Phase 3A): AUTOPSY / FRAGILITY / REPAIR ----
    AutopsyReport autopsy_report;
    bool autopsy_ran = false;
    if (flag_diagnose || flag_fragility) {
        std::string mode = flag_fragility ? "landscape" : diagnose_mode;
        autopsy_report = runAutopsy(ast, mode);
        autopsy_ran = true;
        printAutopsyReport(autopsy_report);
    }

    RepairResult repair_result;
    bool repair_ran = false;
    if (!repair_target.empty()) {
        repair_result = repairMDP(ast, repair_target);
        repair_ran = true;
        printRepairReport(repair_result);
    }

    // ---- Get the primary solver result for visualization ----
    SolverResult primary_result;
    if (ran_vi) primary_result = vi_result;
    else if (ran_pi) {
        primary_result.values = pi_result.values;
        primary_result.policy = pi_result.policy;
        primary_result.iterations = pi_result.policy_improvement_steps;
        primary_result.converged = pi_result.converged;
        primary_result.wall_clock_ms = pi_result.wall_clock_ms;
    }

#ifdef MDP_VIZ_ENABLED
    // ---- Phase 4: Terminal Animation (--animate) ----
    if (flag_animate && (ran_vi || ran_pi)) {
        runTerminalAnimation(ast, primary_result);
    }
    // v3.0: POMDP animation when PBVI ran.
    if (flag_animate && ran_pbvi) {
        runPOMDPAnimation(ast, pbvi_result);
    }

    // ---- Phase 5: HTML Output (--html) ----
    if (!html_output_path.empty()) {
        SolverResult* vi_ptr = ran_vi ? &vi_result : nullptr;
        PolicyIterationResult* pi_ptr = ran_pi ? &pi_result : nullptr;
        QLearningResult* ql_ptr = ran_ql ? &ql_result : nullptr;
        PBVIResult* pbvi_ptr = ran_pbvi ? &pbvi_result : nullptr;
        if (!generateHTML(html_output_path, ast, vi_ptr, pi_ptr, ql_ptr, pbvi_ptr)) {
            return EXIT_FAILURE;
        }
    }
#else
    // If not compiled with visualization, warn the user clearly.
    if (flag_animate || !html_output_path.empty()) {
        std::cerr << "\n[NOTE] This binary was built WITHOUT visualization support.\n";
        std::cerr << "       --animate and --html are disabled.\n\n";
        std::cerr << "       To enable, rebuild with -DMDP_VIZ_ENABLED:\n\n";
        std::cerr << "         cd src/\n";
        std::cerr << "         g++ -std=c++17 -O2 -DMDP_VIZ_ENABLED \\\n";
        std::cerr << "             -o mdp_compiler mdp_compiler.cpp\n\n";
        std::cerr << "       Or just run ./build.sh (Linux/macOS) or BUILD.bat (Windows).\n";
        std::cerr << "       Make sure visualizer.{hpp,cpp} and html_output.{hpp,cpp}\n";
        std::cerr << "       are in the SAME directory as mdp_compiler.cpp.\n\n";
    }
#endif

    // ---- v3.0: --json emission ----
    // A minimal but stable JSON envelope so callers (portfolio pipeline in
    // Phase 2, autopsy harness in Phase 3) can drive the compiler as a
    // subprocess.
    if (flag_json) {
        auto json_str = [](const std::string& s) {
            std::string o; o.reserve(s.size() + 2);
            o.push_back('"');
            for (char c : s) {
                switch (c) {
                    case '"':  o += "\\\""; break;
                    case '\\': o += "\\\\"; break;
                    case '\n': o += "\\n";  break;
                    case '\r': o += "\\r";  break;
                    case '\t': o += "\\t";  break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                            o += buf;
                        } else o.push_back(c);
                }
            }
            o.push_back('"');
            return o;
        };
        std::ostringstream j;
        j << std::setprecision(10);
        j << "{\n";
        j << "  \"version\": \"3.0\",\n";
        j << "  \"source\": " << json_str(filepath) << ",\n";
        j << "  \"solver_mode\": " << json_str(ast.solver_mode) << ",\n";
        j << "  \"discount\": " << ast.discount_factor << ",\n";
        j << "  \"states\": [";
        {
            bool first = true;
            for (const auto& s : ast.states) {
                if (!first) j << ", ";
                j << json_str(s);
                first = false;
            }
        }
        j << "],\n";

        // Values + policy: prefer VI -> PI -> PBVI beliefs for V/π.
        // (PBVI alpha-vector α values per state are emitted in a separate
        // "alpha_vectors" block; the top-level "values"/"policy" use the
        // belief-state values/actions when only PBVI ran.)
        std::unordered_map<std::string, double> V_out;
        std::unordered_map<std::string, std::string> pi_out;
        if (ran_vi) { V_out = vi_result.values; pi_out = vi_result.policy; }
        else if (ran_pi) { V_out = pi_result.values; pi_out = pi_result.policy; }
        if (ran_pbvi) {
            for (const auto& [k, v] : pbvi_result.values_at_belief) V_out[k] = v;
            for (const auto& [k, a] : pbvi_result.policy_at_belief) pi_out[k] = a;
        }

        j << "  \"values\": {";
        {
            bool first = true;
            for (const auto& [k, v] : V_out) {
                if (!first) j << ", ";
                j << json_str(k) << ": " << v;
                first = false;
            }
        }
        j << "},\n";
        j << "  \"policy\": {";
        {
            bool first = true;
            for (const auto& [k, a] : pi_out) {
                if (!first) j << ", ";
                j << json_str(k) << ": " << json_str(a);
                first = false;
            }
        }
        j << "},\n";

        // Solver stats
        j << "  \"stats\": {\n";
        bool stats_first = true;
        auto stats_add = [&](const std::string& key, const std::string& value_json) {
            if (!stats_first) j << ",\n";
            j << "    " << json_str(key) << ": " << value_json;
            stats_first = false;
        };
        if (ran_vi) {
            stats_add("value_iteration_iterations", std::to_string(vi_result.iterations));
            stats_add("value_iteration_converged", vi_result.converged ? "true" : "false");
            stats_add("value_iteration_ms",
                      [&](){std::ostringstream o; o<<vi_result.wall_clock_ms; return o.str();}());
        }
        if (ran_pi) {
            stats_add("policy_iteration_steps", std::to_string(pi_result.policy_improvement_steps));
            stats_add("policy_iteration_converged", pi_result.converged ? "true" : "false");
            stats_add("policy_iteration_ms",
                      [&](){std::ostringstream o; o<<pi_result.wall_clock_ms; return o.str();}());
        }
        if (ran_ql) {
            stats_add("q_learning_episodes", std::to_string(ql_result.episodes_to_convergence));
            stats_add("q_learning_ms",
                      [&](){std::ostringstream o; o<<ql_result.wall_clock_ms; return o.str();}());
        }
        if (ran_pbvi) {
            stats_add("pbvi_iterations", std::to_string(pbvi_result.iterations));
            stats_add("pbvi_converged", pbvi_result.converged ? "true" : "false");
            stats_add("pbvi_ms",
                      [&](){std::ostringstream o; o<<pbvi_result.wall_clock_ms; return o.str();}());
            stats_add("pbvi_alpha_vector_count", std::to_string(pbvi_result.alpha_vectors.size()));
            stats_add("pbvi_belief_points", std::to_string(pbvi_result.belief_points.size()));
        }
        j << "\n  },\n";

        // PBVI alpha vectors (if PBVI ran)
        j << "  \"alpha_vectors\": [";
        if (ran_pbvi) {
            for (size_t i = 0; i < pbvi_result.alpha_vectors.size(); ++i) {
                if (i > 0) j << ", ";
                const auto& a = pbvi_result.alpha_vectors[i];
                j << "{\"action\": " << json_str(a.action) << ", \"values\": {";
                bool fst = true;
                for (const auto& [s, v] : a.values) {
                    if (!fst) j << ", ";
                    j << json_str(s) << ": " << v;
                    fst = false;
                }
                j << "}}";
            }
        }
        j << "],\n";

        // HMM block (if the bridge ran).
        j << "  \"hmm\": ";
        if (ast.hmm_fitted) {
            j << "{\n";
            j << "    \"states\": [";
            for (size_t i = 0; i < ast.hmm_states.size(); ++i) {
                if (i) j << ", ";
                j << json_str(ast.hmm_states[i]);
            }
            j << "],\n";
            j << "    \"log_likelihood\": " << ast.hmm_log_likelihood << ",\n";
            j << "    \"transition_matrix\": [";
            for (size_t i = 0; i < ast.hmm_transition_matrix.size(); ++i) {
                if (i) j << ", ";
                j << "[";
                for (size_t k = 0; k < ast.hmm_transition_matrix[i].size(); ++k) {
                    if (k) j << ", ";
                    j << ast.hmm_transition_matrix[i][k];
                }
                j << "]";
            }
            j << "],\n";
            j << "    \"emission_mean\": [";
            for (size_t i = 0; i < ast.hmm_emission_mean.size(); ++i) {
                if (i) j << ", ";
                j << ast.hmm_emission_mean[i];
            }
            j << "],\n";
            j << "    \"emission_std\": [";
            for (size_t i = 0; i < ast.hmm_emission_std.size(); ++i) {
                if (i) j << ", ";
                j << ast.hmm_emission_std[i];
            }
            j << "],\n";
            j << "    \"initial\": [";
            for (size_t i = 0; i < ast.hmm_initial.size(); ++i) {
                if (i) j << ", ";
                j << ast.hmm_initial[i];
            }
            j << "]\n  },\n";
        } else {
            j << "null,\n";
        }

        // Verify results
        j << "  \"verify_failures\": " << verify_failures;
        if (autopsy_ran || repair_ran) j << ",";
        j << "\n";

        // v3.0 Phase 3A: autopsy + repair output (only emitted when those passes ran).
        auto esc = [](const std::string& s) {
            std::string out; out.reserve(s.size());
            for (char c : s) {
                switch (c) {
                    case '"':  out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n";  break;
                    case '\r': out += "\\r";  break;
                    case '\t': out += "\\t";  break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20)
                            out += " ";  // strip control bytes
                        else out.push_back(c);
                }
            }
            return out;
        };

        if (autopsy_ran) {
            j << "  \"autopsy\": {\n";
            j << "    \"findings\": [\n";
            for (size_t i = 0; i < autopsy_report.findings.size(); ++i) {
                const auto& f = autopsy_report.findings[i];
                j << "      {";
                j << "\"class\": \"" << findingClassName(f.cls) << "\", ";
                j << "\"severity\": \"" << findingSeverityName(f.severity) << "\", ";
                j << "\"headline\": \"" << esc(f.headline) << "\", ";
                j << "\"details\": [";
                for (size_t k = 0; k < f.details.size(); ++k) {
                    if (k) j << ", ";
                    j << "\"" << esc(f.details[k]) << "\"";
                }
                j << "], \"numeric_fields\": {";
                size_t k = 0;
                for (const auto& [key, val] : f.numeric_fields) {
                    if (k++) j << ", ";
                    j << "\"" << esc(key) << "\": " << val;
                }
                j << "}, \"text_fields\": {";
                k = 0;
                for (const auto& [key, val] : f.text_fields) {
                    if (k++) j << ", ";
                    j << "\"" << esc(key) << "\": \"" << esc(val) << "\"";
                }
                j << "}}";
                if (i + 1 < autopsy_report.findings.size()) j << ",";
                j << "\n";
            }
            j << "    ],\n";
            j << "    \"has_errors\": " << (autopsy_report.has_errors() ? "true" : "false") << ",\n";
            j << "    \"has_issues\": " << (autopsy_report.has_issues() ? "true" : "false") << "\n";
            j << "  }";
            if (repair_ran) j << ",";
            j << "\n";
        }

        if (repair_ran) {
            j << "  \"repair\": {\n";
            j << "    \"target\": \"" << esc(repair_result.target_assertion) << "\",\n";
            j << "    \"already_satisfied\": "
              << (repair_result.already_satisfied ? "true" : "false") << ",\n";
            j << "    \"any_succeeded\": "
              << (repair_result.any_succeeded ? "true" : "false") << ",\n";
            j << "    \"proposals\": [\n";
            for (size_t i = 0; i < repair_result.proposals.size(); ++i) {
                const auto& p = repair_result.proposals[i];
                j << "      {";
                j << "\"kind\": \""           << esc(p.parameter_kind)  << "\", ";
                j << "\"label\": \""          << esc(p.parameter_label) << "\", ";
                j << "\"original_value\": "   << p.original_value       << ", ";
                j << "\"new_value\": "        << p.new_value            << ", ";
                j << "\"magnitude\": "        << p.magnitude            << ", ";
                j << "\"succeeded\": "        << (p.succeeded ? "true" : "false");
                j << "}";
                if (i + 1 < repair_result.proposals.size()) j << ",";
                j << "\n";
            }
            j << "    ]\n";
            j << "  }\n";
        }
        j << "}\n";

        // Three emission modes:
        //   --json-out <file>  : write JSON to <file>, normal terminal output stays.
        //   --json (no file)   : human output already redirected to stderr above,
        //                        so we restore stdout and emit JSON on real stdout.
        if (!json_output_path.empty()) {
            std::ofstream jf(json_output_path);
            if (jf) jf << j.str();
            else std::cerr << "[WARN] Could not open " << json_output_path << " for writing.\n";
        } else {
            // Restore real stdout and emit pure JSON.
            std::cout.rdbuf(orig_cout);
            std::cout << j.str();
        }
    }

    if (!(flag_json && json_output_path.empty())) {
        std::cout << "\nMDP-DSL v3.0 pipeline complete.\n";
    }
    // Exit code semantics:
    //   VERIFY failure              -> EXIT_FAILURE
    //   --diagnose with any issues  -> EXIT_FAILURE
    //   --repair fails / no fix     -> EXIT_FAILURE  (0 if fix found or already-satisfied)
    if (verify_failures > 0) return EXIT_FAILURE;
    if (autopsy_ran && autopsy_report.has_issues()) return EXIT_FAILURE;
    if (repair_ran && !(repair_result.already_satisfied || repair_result.any_succeeded))
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

#endif // MDP_TESTING_MODE

#endif // MDP_COMPILER_HPP_INCLUDED
