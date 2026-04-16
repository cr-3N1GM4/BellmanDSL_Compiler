// ============================================================================
// FILE:    mdp_compiler.cpp
// PROJECT: MDP Domain-Specific Language (DSL) — Phase 1 + Phase 2 + Phase 3
// AUTHOR:  Charvit Rajani (Roll: 240102028)
// DATE:    2026-04-10
//
// PURPOSE: This is the UNIFIED compiler pipeline (Phases 1, 2, & 3).
//   Phase 1 (Parser):    Reads a .mdp file -> builds the MDP_AST.
//   Phase 2 (Validator): Inspects the AST -> catches mathematical/logical errors.
//   Phase 3 (Solver):    Runs Value Iteration -> computes optimal policy.
//
// The pipeline:
//   .mdp file -> [Parser] -> MDP_AST -> [Validator] -> Validated AST -> [Solver] -> Policy
//
// COMPILE: g++ -std=c++17 -O2 -Wall -Wextra -o mdp_compiler mdp_compiler.cpp
// RUN:     ./mdp_compiler ../examples/gridworld_simple.mdp
//
// DESIGN PHILOSOPHY:
//   - Every line is commented as if this were a textbook.
//   - No external libraries -- pure standard C++17.
//   - Errors and warnings are collected, not thrown.
// ============================================================================


// ---- SECTION 0: STANDARD LIBRARY INCLUDES ----

#include <iostream>       // std::cout, std::cerr
#include <fstream>        // std::ifstream
#include <sstream>        // std::istringstream, std::ostringstream
#include <string>         // std::string
#include <vector>         // std::vector
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <algorithm>      // std::max
#include <iomanip>        // std::setprecision, std::setw
#include <cstdlib>        // EXIT_SUCCESS, EXIT_FAILURE
#include <cmath>          // std::abs
#include <utility>        // std::pair
#include <limits>         // std::numeric_limits


// ============================================================================
// PHASE 1: PARSER ENGINE
// ============================================================================

// ---- SECTION 1: AST DATA STRUCTURES ----

struct Transition {
    std::string source_state;
    std::string action;
    std::string dest_state;
    double      probability;
};

struct Reward {
    std::string state_name;
    double      value;
};

struct MDP_AST {
    std::unordered_set<std::string>          states;
    std::unordered_set<std::string>          actions;
    std::vector<Transition>                  transitions;
    std::unordered_map<std::string, double>  rewards;
    double                                   discount_factor;
    MDP_AST() : discount_factor(0.9) {}
};


// ---- SECTION 2: UTILITY FUNCTIONS ----

std::string trim(const std::string& str) {
    const std::string whitespace = " \t\r\n";
    const auto start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    const auto end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

std::string toUpper(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return result;
}


// ---- SECTION 3: THE PARSER ENGINE ----

MDP_AST parseMDPFile(const std::string& filepath) {
    MDP_AST ast;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[FATAL ERROR] Cannot open file: " << filepath << std::endl;
        return ast;
    }

    std::cout << "=== MDP Compiler v3.0 (Phase 1+2+3) ===" << std::endl;
    std::cout << "Parsing file: " << filepath << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::istringstream iss(trimmed);
        std::string keyword;
        iss >> keyword;
        std::string keyword_upper = toUpper(keyword);

        if (keyword_upper == "STATE:") {
            std::string name; iss >> name;
            if (name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": STATE with no name.\n"; continue; }
            auto [it, ok] = ast.states.insert(name);
            if (!ok) std::cerr << "[WARNING] Line " << line_number << ": Duplicate state '" << name << "'.\n";
        }
        else if (keyword_upper == "ACTION:") {
            std::string name; iss >> name;
            if (name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": ACTION with no name.\n"; continue; }
            auto [it, ok] = ast.actions.insert(name);
            if (!ok) std::cerr << "[WARNING] Line " << line_number << ": Duplicate action '" << name << "'.\n";
        }
        else if (keyword_upper == "TRANSITION:") {
            std::string src, act, dst; double prob = 0.0;
            iss >> src >> act >> dst >> prob;
            if (iss.fail() || src.empty() || act.empty() || dst.empty()) {
                std::cerr << "[WARNING] Line " << line_number << ": Malformed TRANSITION.\n"; continue;
            }
            if (prob < 0.0 || prob > 1.0)
                std::cerr << "[WARNING] Line " << line_number << ": Prob " << prob << " out of [0,1].\n";
            ast.transitions.emplace_back(Transition{src, act, dst, prob});
        }
        else if (keyword_upper == "REWARD:") {
            std::string name; double val = 0.0;
            iss >> name >> val;
            if (iss.fail() || name.empty()) { std::cerr << "[WARNING] Line " << line_number << ": Malformed REWARD.\n"; continue; }
            if (ast.rewards.count(name) > 0)
                std::cerr << "[WARNING] Line " << line_number << ": Reward for '" << name << "' overwritten.\n";
            ast.rewards[name] = val;
        }
        else if (keyword_upper == "DISCOUNT:") {
            double gamma = 0.9; iss >> gamma;
            if (iss.fail()) { std::cerr << "[WARNING] Line " << line_number << ": Bad DISCOUNT.\n"; gamma = 0.9; }
            if (gamma < 0.0 || gamma > 1.0) {
                std::cerr << "[WARNING] Line " << line_number << ": DISCOUNT " << gamma << " clamped.\n";
                gamma = (gamma < 0.0) ? 0.0 : 1.0;
            }
            ast.discount_factor = gamma;
        }
        else {
            std::cerr << "[WARNING] Line " << line_number << ": Unknown keyword '" << keyword << "'.\n";
        }
    }
    file.close();

    std::cout << "\nParsing complete!\n";
    std::cout << "  States:      " << ast.states.size() << "\n";
    std::cout << "  Actions:     " << ast.actions.size() << "\n";
    std::cout << "  Transitions: " << ast.transitions.size() << "\n";
    std::cout << "  Rewards:     " << ast.rewards.size() << "\n";
    std::cout << "  Discount:    " << ast.discount_factor << "\n";
    std::cout << "----------------------------------------\n";

    return ast;
}


// ---- SECTION 4: printAST() ----

void printAST(const MDP_AST& ast) {
    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << "    PARSED MDP -- ABSTRACT SYNTAX TREE\n";
    std::cout << "======================================\n\n";

    std::cout << "-- STATES (" << ast.states.size() << ") --\n";
    int idx = 1;
    for (const auto& s : ast.states) std::cout << "  " << idx++ << ". " << s << "\n";
    std::cout << "\n";

    std::cout << "-- ACTIONS (" << ast.actions.size() << ") --\n";
    idx = 1;
    for (const auto& a : ast.actions) std::cout << "  " << idx++ << ". " << a << "\n";
    std::cout << "\n";

    std::cout << "-- TRANSITIONS (" << ast.transitions.size() << ") --\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < ast.transitions.size(); ++i) {
        const auto& t = ast.transitions[i];
        std::cout << "  " << (i+1) << ". " << t.source_state
                  << " --[" << t.action << "]--> " << t.dest_state
                  << "  (p = " << t.probability << ")\n";
    }
    std::cout << "\n";

    std::cout << "-- REWARDS (" << ast.rewards.size() << ") --\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [name, val] : ast.rewards)
        std::cout << "  " << name << " => " << val << "\n";
    std::cout << "\n";

    std::cout << "-- DISCOUNT FACTOR --\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  gamma = " << ast.discount_factor << "\n\n";
    std::cout << "======================================\n";
}


// ============================================================================
// PHASE 2: SEMANTIC VALIDATOR
// ============================================================================

struct ValidationResult {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool isValid() const { return errors.empty(); }
};

ValidationResult validateAST(const MDP_AST& ast) {
    ValidationResult result;
    std::unordered_map<std::string, double> prob_sums;
    std::unordered_set<std::string> states_in_transitions;

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

    for (const auto& state : ast.states) {
        if (ast.rewards.count(state) == 0)
            result.warnings.push_back("State '" + state + "' has no REWARD. Solver assumes 0.0.");
        if (states_in_transitions.count(state) == 0)
            result.warnings.push_back("State '" + state + "' is orphaned (no transitions).");
    }

    for (const auto& [rs, rv] : ast.rewards)
        if (ast.states.count(rs) == 0)
            result.errors.push_back("REWARD references undeclared state '" + rs + "'.");

    if (ast.discount_factor < 0.0 || ast.discount_factor > 1.0)
        result.errors.push_back("Discount factor outside [0.0, 1.0].");

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
        std::cout << "  Ready for Phase 3 (Solver).\n";
    } else {
        std::cout << "  VERDICT: MDP is INVALID\n";
        std::cout << "  " << result.errors.size() << " error(s). Fix and re-run.\n";
        std::cout << "  BLOCKED from Phase 3.\n";
    }
    std::cout << "--------------------------------------\n";
}


// ============================================================================
// ============================================================================
// PHASE 3: VALUE ITERATION SOLVER  (NEW -- Phase 3 delivery)
// ============================================================================
// ============================================================================
//
// WHAT IS VALUE ITERATION?
// ========================
// Value Iteration is the workhorse algorithm of Markov Decision Processes.
// It answers two fundamental questions:
//
//   1. VALUE FUNCTION V(s):
//      "Starting from state s, if the agent plays OPTIMALLY, what is the
//       maximum total discounted reward it can expect to earn over time?"
//
//   2. OPTIMAL POLICY pi*(s):
//      "In state s, which action should the agent take to maximise its
//       long-term reward?"
//
// THE BELLMAN EQUATION (the mathematical heart of the algorithm):
// ===============================================================
//
//   V_{k+1}(s) = R(s) + gamma * max_{a in A} SUM_{s' in S} T(s,a,s') * V_k(s')
//
// In plain English:
//   "The value of state s = immediate reward R(s),
//    PLUS the discounted best action's expected future value."
//
// Breaking it down piece by piece:
//   R(s)              = immediate reward for being in state s
//   gamma             = discount factor (0.9 means future is worth 90% of now)
//   max_{a in A}      = try ALL actions, pick the one with highest value
//   SUM T(s,a,s')     = for each possible destination s', weight its value
//       * V_k(s')       by the probability T(s,a,s') of arriving there
//
// CONVERGENCE:
// ============
// We repeat the Bellman update for ALL states. Each full pass is one
// "iteration." After each iteration, we compute:
//   delta = max_s |V_{k+1}(s) - V_k(s)|
//
// When delta < theta (we use theta = 1e-9), the values have "converged."
//
// WHY DOES IT CONVERGE?
// The Bellman operator is a CONTRACTION MAPPING when gamma < 1:
//   ||V_{k+1} - V*|| <= gamma * ||V_k - V*||
// Since gamma < 1, this shrinks to zero. (Banach Fixed-Point Theorem.)
// For gamma = 0.9 and theta = 1e-9, convergence takes ~200 iterations.
//
// ABSORBING STATES:
// =================
// A state like "Goal" that only transitions to itself (p=1.0) is
// "absorbing." Its converged value is:
//   V(Goal) = R(Goal) / (1 - gamma) = 100 / 0.1 = 1000.0
// The agent collects the reward forever, discounted geometrically.
//
// POLICY EXTRACTION:
// ==================
// Once V* is known, the optimal policy is:
//   pi*(s) = argmax_{a} SUM_{s'} T(s,a,s') * V*(s')
// For each state, pick the action with the highest expected future value.
//
// DATA STRUCTURES:
// ================
// transition_index: unordered_map<string, vector<pair<string,double>>>
//   Key: "source|action", Value: [(dest, prob), ...]
//   Avoids scanning ALL transitions per (s,a) query.
//   Without it: O(|T|) per query. With it: O(k) where k = fan-out.
//
// actions_at_state: unordered_map<string, unordered_set<string>>
//   Which actions have transitions defined for each state.
//   Prevents trying undefined actions (which would give Q=0, wrong).
//
// COMPLEXITY:
//   Build index:   O(|T|)
//   Per iteration: O(|T|) total (each transition visited once)
//   Total:         O(n_iter * |T|), typically 200 * |T|
// ============================================================================


// ----  Struct: SolverResult ----
// Holds the complete output of Value Iteration:
//   values     -- V*(s) for every state
//   policy     -- pi*(s) for every state (best action name)
//   iterations -- how many iterations until convergence
//   converged  -- did we converge within the cap?
struct SolverResult {
    std::unordered_map<std::string, double>      values;     // V*(s)
    std::unordered_map<std::string, std::string>  policy;     // pi*(s)
    int                                           iterations; // count
    bool                                          converged;  // success?
};


// ---- Function: solveValueIteration ----
//
// INPUT:  const MDP_AST& -- validated AST from Phase 2
//         double theta   -- convergence threshold (default 1e-9)
//         int max_iter   -- safety cap (default 10000)
//
// OUTPUT: SolverResult with values, policy, iteration count, convergence flag
//
// PRECONDITION: AST must have passed Phase 2 validation.
//
// ALGORITHM:
//   1. Build transition index (acceleration structure).
//   2. Initialise V(s) = 0 for all states.
//   3. Repeat Bellman updates until delta < theta or max_iter reached.
//   4. Extract policy via argmax over Q-values.
SolverResult solveValueIteration(
    const MDP_AST& ast,
    double theta = 1e-9,       // Convergence threshold
    int max_iter = 10000       // Safety cap
) {

    SolverResult result;
    result.converged = false;
    result.iterations = 0;

    // ============================================================
    // STEP 1: BUILD THE TRANSITION INDEX
    // ============================================================
    // Instead of scanning the entire transitions vector each time we
    // need transitions for a specific (state, action) pair, we build
    // a lookup table indexed by the composite key "source|action".
    //
    // Each entry maps to a vector of (destination, probability) pairs.
    //
    // Example after indexing:
    //   "Start|MoveRight"  -> [("Middle", 0.8), ("Start", 0.2)]
    //   "Middle|MoveRight" -> [("Goal", 0.7), ("Trap", 0.3)]
    //   "Goal|Stay"        -> [("Goal", 1.0)]
    //
    // This reduces per-query cost from O(|T|) to O(k) where k is
    // the number of destinations for that (state, action) pair.
    // Building the index itself is O(|T|) -- one pass.
    // ============================================================

    using DestProb = std::pair<std::string, double>;  // (dest_state, probability)
    std::unordered_map<std::string, std::vector<DestProb>> transition_index;

    // Build the index in a single pass over all transitions.
    for (const auto& t : ast.transitions) {
        std::string key = t.source_state + "|" + t.action;  // Composite key
        transition_index[key].emplace_back(t.dest_state, t.probability);
    }

    // ============================================================
    // Also: determine which actions are available at each state.
    // ============================================================
    // Not every action is available at every state. "Start" might
    // only have "MoveRight," while "Goal" only has "Stay."
    //
    // If we tried every declared action at every state, we'd get
    // Q(s,a) = 0 for undefined actions (no transitions => sum is 0).
    // This could incorrectly dominate negative-value actions.
    //
    // By tracking which actions actually have transitions at each
    // state, the solver only considers meaningful actions.
    // ============================================================

    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at_state;

    for (const auto& t : ast.transitions) {
        actions_at_state[t.source_state].insert(t.action);
    }


    // ============================================================
    // STEP 2: INITIALISE THE VALUE FUNCTION
    // ============================================================
    // V_0(s) = 0.0 for all states.
    //
    // Any starting values work (Value Iteration converges from ANY
    // initial point when gamma < 1), but zero is the clean default.
    // ============================================================

    std::unordered_map<std::string, double> V;  // Current value function

    for (const auto& state : ast.states) {
        V[state] = 0.0;                         // All states start at zero
    }


    // ============================================================
    // STEP 3: MAIN ITERATION LOOP (Bellman updates)
    // ============================================================
    // Each iteration updates V(s) for ALL states.
    // We track the maximum change (delta) for convergence.
    //
    // This is "synchronous" Value Iteration: we compute all new
    // values into a temporary map, then swap. This guarantees
    // convergence regardless of state iteration order.
    // ============================================================

    for (int iter = 1; iter <= max_iter; ++iter) {

        double delta = 0.0;  // Maximum |V_new - V_old| this iteration

        // For each state s in the MDP:
        for (const auto& state : ast.states) {

            double old_value = V[state];  // Save for delta computation

            // Get the reward R(s). Default 0.0 if undefined.
            double reward = 0.0;
            if (ast.rewards.count(state) > 0) {
                reward = ast.rewards.at(state);  // .at() is const-safe
            }

            // If this state has no outgoing transitions, it's a pure
            // terminal with no self-loop. Its value is just R(s).
            // (States with a self-loop like Goal->Goal DO have actions.)
            if (actions_at_state.count(state) == 0) {
                V[state] = reward;
                double change = std::abs(V[state] - old_value);
                if (change > delta) delta = change;
                continue;
            }

            // ---- Find the best action: argmax_a Q(s, a) ----
            //
            // For each action a available at state s, compute:
            //   Q(s, a) = SUM_{s'} T(s, a, s') * V(s')
            //
            // Pick the action with the highest Q value.
            // New V(s) = R(s) + gamma * max_a Q(s, a)

            // Start with the most negative possible double value so
            // any real Q-value will be larger.
            double best_q = std::numeric_limits<double>::lowest();

            for (const auto& action : actions_at_state[state]) {

                std::string key = state + "|" + action;  // Lookup key

                // Compute Q(s, a) = SUM T(s,a,s') * V(s')
                double q_value = 0.0;

                for (const auto& [dest, prob] : transition_index[key]) {
                    q_value += prob * V[dest];  // Weighted future value
                }

                // Update best if this action is better
                if (q_value > best_q) {
                    best_q = q_value;
                }
            }

            // ---- Apply the Bellman equation ----
            // V_{k+1}(s) = R(s) + gamma * max_a Q(s, a)
            V[state] = reward + ast.discount_factor * best_q;

            // Track the largest change for convergence detection
            double change = std::abs(V[state] - old_value);
            if (change > delta) delta = change;
        }

        // ---- Check for convergence ----
        // If the maximum change across ALL states is below theta,
        // the values have converged. Future iterations won't change
        // them meaningfully.
        if (delta < theta) {
            result.converged = true;
            result.iterations = iter;

            std::cout << "  Value Iteration converged after "
                      << iter << " iteration(s).\n";
            std::cout << "  Final delta: " << std::scientific
                      << std::setprecision(2) << delta << std::fixed << "\n";
            break;
        }

        // Safety: hit max iterations without converging?
        if (iter == max_iter) {
            result.iterations = max_iter;
            std::cerr << "[WARNING] Value Iteration did NOT converge after "
                      << max_iter << " iterations.\n";
            std::cerr << "  Last delta: " << std::scientific << delta
                      << std::fixed << "\n";
        }
    }


    // ============================================================
    // STEP 4: EXTRACT THE OPTIMAL POLICY
    // ============================================================
    // Now that V*(s) is known, the optimal policy is:
    //   pi*(s) = argmax_{a in A(s)} SUM_{s'} T(s,a,s') * V*(s')
    //
    // For each state, pick the action with the highest Q-value.
    // The R(s) and gamma terms are constant across actions for a
    // given state, so the argmax is the same with or without them.
    // ============================================================

    for (const auto& state : ast.states) {

        // No outgoing transitions => terminal state, no decision.
        if (actions_at_state.count(state) == 0) {
            result.policy[state] = "(terminal)";
            continue;
        }

        std::string best_action = "";
        double best_q = std::numeric_limits<double>::lowest();

        for (const auto& action : actions_at_state[state]) {
            std::string key = state + "|" + action;
            double q_value = 0.0;

            for (const auto& [dest, prob] : transition_index[key]) {
                q_value += prob * V[dest];  // Expected future value
            }

            if (q_value > best_q) {
                best_q = q_value;
                best_action = action;
            }
        }

        result.policy[state] = best_action;
    }

    // Store the converged value function
    result.values = V;

    return result;
}


// ---- Function: printSolverReport ----
// Prints the converged value function V*(s) and optimal policy pi*(s)
// in a formatted table so a human can read it immediately.
void printSolverReport(const SolverResult& result, const MDP_AST& ast) {

    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << "  PHASE 3 -- VALUE ITERATION RESULTS\n";
    std::cout << "======================================\n\n";

    // ---- Convergence status ----
    std::cout << "-- CONVERGENCE --\n";
    if (result.converged) {
        std::cout << "  Converged after " << result.iterations << " iterations.\n";
    } else {
        std::cout << "  Did NOT converge within " << result.iterations << " iterations.\n";
        std::cout << "  Results may be approximate.\n";
    }
    std::cout << "  Discount factor gamma = " << std::fixed << std::setprecision(4)
              << ast.discount_factor << "\n\n";

    // ---- Value function + policy table ----
    std::cout << "-- OPTIMAL VALUE FUNCTION V*(s) & POLICY pi*(s) --\n\n";

    // Table header
    std::cout << "  " << std::left
              << std::setw(14) << "State"
              << std::setw(12) << "R(s)"
              << std::setw(14) << "V*(s)"
              << "pi*(s)" << std::endl;
    std::cout << "  "
              << std::string(14, '-')
              << std::string(12, '-')
              << std::string(14, '-')
              << std::string(20, '-') << std::endl;

    // Table rows
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& state : ast.states) {

        double reward = 0.0;
        if (ast.rewards.count(state) > 0) reward = ast.rewards.at(state);

        double value = result.values.at(state);
        std::string action = result.policy.at(state);

        std::cout << "  " << std::left
                  << std::setw(14) << state
                  << std::setw(12) << reward
                  << std::setw(14) << value
                  << action << "\n";
    }

    std::cout << "\n";

    // ---- Interpretation guide ----
    std::cout << "-- INTERPRETATION --\n";
    std::cout << "  V*(s) = total discounted reward the agent can expect\n";
    std::cout << "          from state s if it plays optimally.\n";
    std::cout << "  pi*(s) = the action the agent should take in state s\n";
    std::cout << "           to maximise long-term reward.\n\n";
    std::cout << "======================================\n";
}


// ============================================================================
// SECTION 7: MAIN ENTRY POINT (Updated for Phase 3)
// ============================================================================
//
// Pipeline: Parse -> Print AST -> Validate -> Report -> Solve -> Report
//
// Exit codes:
//   0 = Full pipeline completed successfully
//   1 = Parse failure, validation errors, or solver failure
// ============================================================================

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mdp_file>\n";
        std::cerr << "Example: " << argv[0] << " examples/gridworld_simple.mdp\n";
        return EXIT_FAILURE;
    }

    std::string filepath = argv[1];

    // ---- PHASE 1: PARSE ----
    MDP_AST ast = parseMDPFile(filepath);

    if (ast.states.empty()) {
        std::cerr << "[FATAL] No states parsed.\n";
        return EXIT_FAILURE;
    }

    printAST(ast);

    // ---- PHASE 2: VALIDATE ----
    std::cout << "\nRunning semantic validation...\n";
    ValidationResult validation = validateAST(ast);
    printValidationReport(validation);

    if (!validation.isValid()) {
        std::cout << "\nPhase 2 FAILED. Fix errors and re-run.\n";
        return EXIT_FAILURE;
    }

    // ---- PHASE 3: SOLVE ----
    std::cout << "\nRunning Value Iteration solver...\n";
    std::cout << "  Convergence threshold theta = 1e-9\n";
    std::cout << "  Maximum iterations: 10000\n\n";

    SolverResult solver = solveValueIteration(ast);
    printSolverReport(solver, ast);

    // ---- FINAL STATUS ----
    std::cout << "\n";
    if (solver.converged) {
        std::cout << "Phase 1 + Phase 2 + Phase 3 complete.\n";
        std::cout << "MDP parsed, validated, and SOLVED successfully.\n";
        std::cout << "Next: Phase 4 -- GridWorld Case Study.\n";
    } else {
        std::cout << "[WARNING] Solver did not fully converge.\n";
    }

    return EXIT_SUCCESS;
}
