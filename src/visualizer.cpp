// ============================================================================
// FILE:    visualizer.cpp
// PURPOSE: Terminal-based ANSI visualization for MDP-DSL.
//          Draws colored grids, animates VI convergence, traces policy paths.
//          Works on any 256-color terminal. Falls back to plain ASCII if not.
// ============================================================================

#include "visualizer.hpp"

// We need access to MDP_AST, SolverResult, getReward from the main compiler.
// These are provided when compiled together as a single unit via the build command.
// Do NOT include mdp_compiler.cpp here — it causes multiple definition errors.
// Instead, the build command compiles everything together.

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <thread>
#include <chrono>
#include <cstdlib>

// ---- terminalSupportsColor: check whether the console supports ANSI ----
// On Windows we enable VT processing in main(), so we can assume color
// support. On POSIX systems we check the TERM env var (false if unset,
// empty, or "dumb").
bool terminalSupportsColor() {
#ifdef _WIN32
    return true;
#else
    const char* term = std::getenv("TERM");
    if (!term || std::string(term).empty() || std::string(term) == "dumb")
        return false;
    return true;
#endif
}

// ---- detectGrid: determine if states follow RrCc naming pattern ----
// Parses each state name with regex R(\d+)C(\d+). If ALL states match,
// builds a grid layout. Otherwise returns is_grid=false.
GridInfo detectGrid(const MDP_AST& ast) {
    GridInfo gi;
    gi.is_grid = false;
    gi.rows = 0;
    gi.cols = 0;

    // Try to parse every state name as RrCc
    std::regex rc_pattern("R(\\d+)C(\\d+)");
    std::smatch match;
    bool all_match = true;

    for (const auto& s : ast.states) {
        if (std::regex_match(s, match, rc_pattern)) {
            int r = std::stoi(match[1].str());
            int c = std::stoi(match[2].str());
            gi.state_to_rc[s] = {r, c};
            gi.rows = std::max(gi.rows, r);
            gi.cols = std::max(gi.cols, c);
        } else {
            all_match = false;
            break;
        }
    }

    if (!all_match || ast.states.size() < 2) {
        gi.is_grid = false;
        return gi;
    }

    gi.is_grid = true;

    // Build 2D grid array (1-indexed internally, stored 0-indexed)
    gi.grid.resize(gi.rows, std::vector<std::string>(gi.cols, ""));
    for (const auto& [name, rc] : gi.state_to_rc) {
        gi.grid[rc.first - 1][rc.second - 1] = name;
    }

    // Find goal (highest reward) and trap (lowest reward)
    double max_r = -1e18, min_r = 1e18;
    for (const auto& [name, val] : ast.rewards) {
        if (val > max_r) { max_r = val; gi.goal_state = name; }
        if (val < min_r) { min_r = val; gi.trap_state = name; }
    }

    return gi;
}

// ---- valueToAnsiColor: map V* to 256-color code (red->yellow->green) ----
// Uses ANSI 256-color palette: 196=red, 226=yellow, 46=green.
// Normalizes value to [0,1] range using v_min and v_max.
int valueToAnsiColor(double value, double v_min, double v_max) {
    if (v_max <= v_min) return 226; // Yellow if no range
    double norm = (value - v_min) / (v_max - v_min); // 0.0 to 1.0
    norm = std::max(0.0, std::min(1.0, norm));

    // Map: 0.0->red(196), 0.5->yellow(226), 1.0->green(46)
    if (norm < 0.5) {
        // Red to yellow: 196 -> 202 -> 208 -> 214 -> 220 -> 226
        int idx = static_cast<int>(norm * 2.0 * 5.0); // 0..5
        return 196 + idx;
    } else {
        // Yellow to green: 226 -> 190 -> 154 -> 118 -> 82 -> 46
        int idx = static_cast<int>((norm - 0.5) * 2.0 * 5.0); // 0..5
        return 226 - idx * 36;
    }
}

// ---- actionArrow: convert action name to Unicode arrow ----
std::string actionArrow(const std::string& action) {
    std::string a = action;
    // Convert to lowercase for matching
    for (auto& c : a) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (a == "up")    return "\xe2\x86\x91"; // ↑
    if (a == "down")  return "\xe2\x86\x93"; // ↓
    if (a == "left")  return "\xe2\x86\x90"; // ←
    if (a == "right") return "\xe2\x86\x92"; // →
    if (a == "stay" || a == "(terminal)") return "\xe2\x97\x8f"; // ●
    if (a == "moveright") return "\xe2\x86\x92";
    if (a == "moveleft")  return "\xe2\x86\x90";
    return action.substr(0, 1); // First char as fallback
}

// ---- clearScreen: ANSI escape to clear terminal and move cursor home ----
static void clearScreen() {
    std::cout << "\033[2J\033[H" << std::flush;
}

// ---- sleepMs: portable millisecond sleep ----
static void sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ---- drawGrid: render the grid with colored cells, arrows, optional cursor ----
// Each cell is a fixed-width box showing state name, V* value, and policy arrow.
// cursor_state highlights one cell with a ● character.
// stats_text is printed to the right of the grid.
void drawGrid(const GridInfo& gi, const MDP_AST& ast,
              const std::unordered_map<std::string, double>& values,
              const std::unordered_map<std::string, std::string>& policy,
              const std::string& cursor_state,
              const std::string& stats_text,
              bool use_color) {

    // Find V* range for color mapping
    double v_min = 1e18, v_max = -1e18;
    for (const auto& [s, v] : values) {
        if (v < v_min) v_min = v;
        if (v > v_max) v_max = v;
    }

    const int CELL_W = 14;  // Characters wide per cell
    const std::string RESET = use_color ? "\033[0m" : "";
    const std::string GRAY_BG = use_color ? "\033[48;5;240m\033[37m" : "";
    const std::string BOLD = use_color ? "\033[1m" : "";

    // Split stats_text into lines for right-side panel
    std::vector<std::string> stats_lines;
    if (!stats_text.empty()) {
        std::istringstream ss(stats_text);
        std::string line;
        while (std::getline(ss, line)) stats_lines.push_back(line);
    }

    std::cout << "\n";

    // Draw row by row, top to bottom (row gi.rows down to row 1)
    int stats_line_idx = 0;
    for (int r = gi.rows; r >= 1; r--) {

        // ---- Top border of row ----
        std::string border_top;
        for (int c = 1; c <= gi.cols; c++) {
            border_top += "+";
            for (int i = 0; i < CELL_W; i++) border_top += "-";
        }
        border_top += "+";
        std::cout << "  " << border_top;
        if (stats_line_idx < (int)stats_lines.size())
            std::cout << "   " << stats_lines[stats_line_idx++];
        std::cout << "\n";

        // ---- Cell content: 3 lines per cell ----
        // Line 1: state name
        // Line 2: V* value
        // Line 3: policy arrow (or cursor)
        for (int line = 0; line < 3; line++) {
            std::string row_str;
            for (int c = 1; c <= gi.cols; c++) {
                std::string state = (r-1 < (int)gi.grid.size() && c-1 < (int)gi.grid[r-1].size())
                                    ? gi.grid[r-1][c-1] : "";

                // Determine cell background color
                std::string bg = "";
                if (use_color) {
                    if (state.empty()) {
                        bg = GRAY_BG; // Wall
                    } else if (state == gi.goal_state) {
                        bg = "\033[48;5;46m\033[30m";  // Bright green bg, black text
                    } else if (state == gi.trap_state) {
                        bg = "\033[48;5;196m\033[37m"; // Bright red bg, white text
                    } else if (values.count(state)) {
                        int col = valueToAnsiColor(values.at(state), v_min, v_max);
                        bg = "\033[48;5;" + std::to_string(col) + "m\033[30m";
                    }
                }

                // Build cell content for this line
                std::string content;
                if (state.empty()) {
                    // Wall cell
                    if (line == 1) content = "  WALL  ";
                    else content = "        ";
                } else {
                    bool is_cursor = (state == cursor_state);
                    if (line == 0) {
                        // State name
                        content = " " + state;
                        while ((int)content.size() < CELL_W - 1) content += " ";
                    } else if (line == 1) {
                        // V* value
                        double v = values.count(state) ? values.at(state) : 0.0;
                        std::ostringstream vs;
                        vs << std::fixed << std::setprecision(2) << v;
                        content = " " + vs.str();
                        while ((int)content.size() < CELL_W - 1) content += " ";
                    } else {
                        // Policy arrow or cursor
                        std::string arrow = policy.count(state) ? actionArrow(policy.at(state)) : "?";
                        if (is_cursor) {
                            content = "   \xe2\x97\x8f     "; // ● cursor
                        } else {
                            content = "   " + arrow + "     ";
                        }
                        while ((int)content.size() < CELL_W - 1) content += " ";
                    }
                    content = content.substr(0, CELL_W - 1);
                }

                // Pad to CELL_W
                while ((int)content.size() < CELL_W) content += " ";

                row_str += "|" + bg + content + RESET;
            }
            row_str += "|";

            std::cout << "  " << row_str;
            if (stats_line_idx < (int)stats_lines.size())
                std::cout << "   " << stats_lines[stats_line_idx++];
            std::cout << "\n";
        }
    }

    // Bottom border
    std::string border_bot;
    for (int c = 1; c <= gi.cols; c++) {
        border_bot += "+";
        for (int i = 0; i < CELL_W; i++) border_bot += "-";
    }
    border_bot += "+";
    std::cout << "  " << border_bot;
    if (stats_line_idx < (int)stats_lines.size())
        std::cout << "   " << stats_lines[stats_line_idx++];
    std::cout << "\n";

    // Print remaining stats lines
    while (stats_line_idx < (int)stats_lines.size()) {
        std::string pad(2 + (CELL_W + 1) * gi.cols + 1, ' ');
        std::cout << pad << "   " << stats_lines[stats_line_idx++] << "\n";
    }
}

// ---- drawLinear: render non-grid MDPs as labeled boxes in a row ----
void drawLinear(const MDP_AST& ast,
                const std::unordered_map<std::string, double>& values,
                const std::unordered_map<std::string, std::string>& policy,
                const std::string& cursor_state,
                bool use_color) {
    const std::string RESET = use_color ? "\033[0m" : "";

    double v_min = 1e18, v_max = -1e18;
    for (const auto& [s, v] : values) {
        if (v < v_min) v_min = v;
        if (v > v_max) v_max = v;
    }

    // Sort states alphabetically for consistent display
    std::vector<std::string> sorted_states(ast.states.begin(), ast.states.end());
    std::sort(sorted_states.begin(), sorted_states.end());

    std::cout << "\n  ";
    for (const auto& s : sorted_states) {
        std::string bg = "";
        if (use_color && values.count(s)) {
            int col = valueToAnsiColor(values.at(s), v_min, v_max);
            bg = "\033[48;5;" + std::to_string(col) + "m\033[30m";
        }
        std::string arrow = policy.count(s) ? actionArrow(policy.at(s)) : "?";
        std::string cursor = (s == cursor_state) ? "\xe2\x97\x8f" : " ";
        double v = values.count(s) ? values.at(s) : 0.0;

        std::ostringstream cell;
        cell << std::fixed << std::setprecision(1);
        cell << "[" << bg << cursor << s << " " << arrow << " V=" << v << RESET << "] ";
        std::cout << cell.str();
    }
    std::cout << "\n\n";
}

// ---- animateValueIteration: show live VI progress every 20 iterations ----
// Re-solves VI from scratch with intermediate display frames.
void animateValueIteration(const MDP_AST& ast, const SolverResult& final_result,
                           const GridInfo& gi, bool use_color) {
    // Build transition index (same as solveValueIteration)
    using DestProb = std::pair<std::string, double>;
    std::unordered_map<std::string, std::vector<DestProb>> tidx;
    std::unordered_map<std::string, std::unordered_set<std::string>> actions_at;
    for (const auto& t : ast.transitions) {
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);
        actions_at[t.source_state].insert(t.action);
    }

    std::unordered_map<std::string, double> V;
    for (const auto& s : ast.states) V[s] = 0.0;

    // Empty policy for intermediate frames
    std::unordered_map<std::string, std::string> empty_policy;
    for (const auto& s : ast.states) empty_policy[s] = "?";

    const int display_interval = 20;
    const int max_iter = final_result.iterations > 0 ? final_result.iterations + 5 : 300;

    for (int iter = 1; iter <= max_iter; ++iter) {
        double delta = 0.0;
        for (const auto& state : ast.states) {
            double old_val = V[state];
            if (actions_at.count(state) == 0) {
                double r = ast.rewards.count(state) ? ast.rewards.at(state) : 0.0;
                V[state] = r;
            } else {
                double best = std::numeric_limits<double>::lowest();
                for (const auto& action : actions_at[state]) {
                    double reward = getReward(ast, state, action);
                    double q = 0.0;
                    for (const auto& [d, p] : tidx[state + "|" + action]) q += p * V[d];
                    double val = reward + ast.discount_factor * q;
                    if (val > best) best = val;
                }
                V[state] = best;
            }
            double ch = std::abs(V[state] - old_val);
            if (ch > delta) delta = ch;
        }

        // Display every N iterations
        if (iter % display_interval == 0 || iter == 1) {
            clearScreen();
            std::ostringstream stats;
            stats << "--- Value Iteration ---\n"
                  << "Iteration: " << iter << "\n"
                  << "Delta: " << std::scientific << std::setprecision(2) << delta << "\n"
                  << std::fixed << "Gamma: " << ast.discount_factor << "\n"
                  << "\nWatching values propagate...";

            if (gi.is_grid) {
                drawGrid(gi, ast, V, empty_policy, "", stats.str(), use_color);
            } else {
                std::cout << stats.str() << "\n";
                drawLinear(ast, V, empty_policy, "", use_color);
            }
            sleepMs(50);
        }

        if (delta < 1e-9) break;
    }

    // Show final converged frame with policy
    sleepMs(500);
    clearScreen();
    std::ostringstream stats;
    stats << "--- Value Iteration ---\n"
          << "CONVERGED!\n"
          << "Iterations: " << final_result.iterations << "\n"
          << "Gamma: " << std::fixed << std::setprecision(2) << ast.discount_factor << "\n"
          << "\nShowing optimal policy...";

    if (gi.is_grid) {
        drawGrid(gi, ast, final_result.values, final_result.policy, "", stats.str(), use_color);
    } else {
        std::cout << stats.str() << "\n";
        drawLinear(ast, final_result.values, final_result.policy, "", use_color);
    }
    sleepMs(2000);
}

// ---- animatePolicyPath: trace the optimal policy from a starting state ----
// Moves a cursor ● through the grid following pi*(s) at each step.
void animatePolicyPath(const MDP_AST& ast, const SolverResult& result,
                       const GridInfo& gi, const std::string& start_state,
                       bool use_color) {
    // Build transition index for deterministic follow
    using DestProb = std::pair<std::string, double>;
    std::unordered_map<std::string, std::vector<DestProb>> tidx;
    for (const auto& t : ast.transitions)
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);

    std::string current = start_state;
    std::unordered_set<std::string> visited;

    for (int step = 0; step < 20; step++) {
        clearScreen();

        std::ostringstream stats;
        stats << "--- Policy Trace ---\n"
              << "Start: " << start_state << "\n"
              << "Step:  " << step << "\n"
              << "At:    " << current << "\n"
              << std::fixed << std::setprecision(2)
              << "V*(" << current << ") = "
              << (result.values.count(current) ? result.values.at(current) : 0.0) << "\n"
              << "Action: " << (result.policy.count(current) ? result.policy.at(current) : "?") << "\n"
              << "\nFollowing pi*(s)...";

        if (gi.is_grid) {
            drawGrid(gi, ast, result.values, result.policy, current, stats.str(), use_color);
        } else {
            std::cout << stats.str() << "\n";
            drawLinear(ast, result.values, result.policy, current, use_color);
        }

        sleepMs(300);

        // Follow the policy to the most probable next state
        if (!result.policy.count(current)) break;
        std::string action = result.policy.at(current);
        if (action == "(terminal)") break;

        std::string key = current + "|" + action;
        if (tidx.count(key) == 0) break;

        // Pick the highest-probability destination (deterministic trace)
        std::string next = current;
        double best_p = -1;
        for (const auto& [d, p] : tidx[key]) {
            if (p > best_p) { best_p = p; next = d; }
        }

        if (visited.count(next) && next == current) break; // Absorbing
        visited.insert(current);
        current = next;
    }

    sleepMs(1000);
}

// ---- runTerminalAnimation: main entry point for --animate flag ----
// Runs: (1) live VI convergence, (2) final policy display, (3) path animations.
void runTerminalAnimation(const MDP_AST& ast, const SolverResult& result) {
    bool use_color = terminalSupportsColor();
    GridInfo gi = detectGrid(ast);

    if (!use_color) {
        std::cout << "[Note] Terminal does not support colors. Using plain ASCII.\n";
    }

    // Phase 1: Animate VI convergence (values propagating)
    std::cout << "\n=== ANIMATION: Value Iteration Convergence ===\n";
    std::cout << "Press Enter to begin...";
    std::cin.get();
    animateValueIteration(ast, result, gi, use_color);

    // Phase 2: Animate policy paths from 3 starting positions
    std::vector<std::string> start_states;

    if (gi.is_grid) {
        // Bottom-left, bottom-right, middle-left
        std::string bl = "R1C1", br = "R1C" + std::to_string(gi.cols);
        std::string ml = "R" + std::to_string((gi.rows + 1) / 2) + "C1";
        if (ast.states.count(bl)) start_states.push_back(bl);
        if (ast.states.count(br)) start_states.push_back(br);
        if (ast.states.count(ml)) start_states.push_back(ml);
    }

    if (start_states.empty()) {
        // For non-grid MDPs, pick up to 3 non-terminal states
        for (const auto& s : ast.states) {
            if (result.policy.count(s) && result.policy.at(s) != "(terminal)") {
                start_states.push_back(s);
                if (start_states.size() >= 3) break;
            }
        }
    }

    for (const auto& start : start_states) {
        animatePolicyPath(ast, result, gi, start, use_color);
    }

    // Final static display
    clearScreen();
    std::ostringstream stats;
    stats << "--- FINAL POLICY ---\n"
          << "Algorithm: Value Iteration\n"
          << "Iterations: " << result.iterations << "\n"
          << "Converged: " << (result.converged ? "Yes" : "No") << "\n"
          << "Gamma: " << std::fixed << std::setprecision(2) << ast.discount_factor << "\n"
          << "States: " << ast.states.size() << "\n"
          << "Transitions: " << ast.transitions.size() << "\n"
          << "\nAnimation complete.";

    if (gi.is_grid) {
        drawGrid(gi, ast, result.values, result.policy, "", stats.str(), use_color);
    } else {
        std::cout << stats.str() << "\n";
        drawLinear(ast, result.values, result.policy, "", use_color);
    }
    std::cout << "\n";
}

// ============================================================================
// v3.0 — POMDP TERMINAL ANIMATION
// ============================================================================
//
// For 2-state POMDPs (the marquee case is Tiger), the belief simplex is the
// segment b(s0) in [0, 1]. We sample the alpha-vector envelope across the
// segment, draw the value curve with 256-color terminal cells, and below it
// stamp the optimal-action region.
//
// We then animate one episode from the initial belief: each step, we sample a
// (deterministic) observation under the current best action, run the Bayes
// filter, and redraw the cursor along the belief axis. This shows the agent
// "thinking" — moving along the simplex as evidence arrives.

void runPOMDPAnimation(const MDP_AST& ast, const PBVIResult& result) {
    bool use_color = terminalSupportsColor();
    if (!use_color)
        std::cout << "[Note] Terminal does not support colors. Using plain ASCII.\n";

    // Sort states deterministically for the simplex axes.
    std::vector<std::string> states(ast.states.begin(), ast.states.end());
    std::sort(states.begin(), states.end());

    std::cout << "\n=== POMDP ANIMATION: PBVI Belief Simplex ===\n";
    std::cout << "Press Enter to begin...";
    std::cin.get();

    if (states.size() != 2) {
        clearScreen();
        std::cout << "[POMDP-ANIMATE] Belief simplex animation is currently 2-state only.\n";
        std::cout << "  Found " << states.size() << " states; printing alpha-vector summary instead.\n\n";
        std::cout << "  Iterations:  " << result.iterations
                  << "    α-vectors: " << result.alpha_vectors.size()
                  << "    Beliefs:   " << result.belief_points.size() << "\n";
        for (const auto& av : result.alpha_vectors) {
            std::cout << "    [" << av.action << "]";
            for (const auto& s : states) {
                double v = av.values.count(s) ? av.values.at(s) : 0.0;
                std::cout << "  " << s << "=" << std::fixed << std::setprecision(2) << v;
            }
            std::cout << "\n";
        }
        return;
    }

    const std::string& s0 = states[0];
    const std::string& s1 = states[1];
    const int N = 60;  // belief-axis resolution (terminal cells)

    // Sample V*(b) and a*(b) along the simplex.
    auto evaluate = [&](double b) -> std::pair<double, std::string> {
        double best = -1e300;
        std::string bestA = "?";
        for (const auto& av : result.alpha_vectors) {
            double v0 = av.values.count(s0) ? av.values.at(s0) : 0.0;
            double v1 = av.values.count(s1) ? av.values.at(s1) : 0.0;
            double val = v0 * b + v1 * (1.0 - b);
            if (val > best) { best = val; bestA = av.action; }
        }
        return {best, bestA};
    };
    std::vector<double> vs(N);
    std::vector<std::string> acts(N);
    double v_min = 1e300, v_max = -1e300;
    for (int i = 0; i < N; ++i) {
        double b = static_cast<double>(i) / (N - 1);
        auto [v, a] = evaluate(b);
        vs[i] = v; acts[i] = a;
        if (v < v_min) v_min = v;
        if (v > v_max) v_max = v;
    }
    if (v_max <= v_min) { v_min -= 1.0; v_max += 1.0; }

    // Map action -> stable ANSI color index for the action-region strip.
    std::unordered_map<std::string, int> action_color;
    static const int palette[] = {196, 220, 46, 39, 99, 208, 51, 207};
    static const int palette_size = sizeof(palette)/sizeof(palette[0]);
    int ci = 0;
    for (const auto& a : acts) {
        if (!action_color.count(a)) { action_color[a] = palette[ci % palette_size]; ci++; }
    }

    // Build the initial belief vector to animate.
    std::unordered_map<std::string, double> belief;
    if (!ast.initial_belief.empty()) {
        double s = 0.0;
        for (const auto& [k, v] : ast.initial_belief) s += v;
        for (const auto& [k, v] : ast.initial_belief) belief[k] = (s > 0 ? v / s : 0.0);
    } else {
        belief[s0] = 0.5; belief[s1] = 0.5;
    }

    // Helper: draw one frame.
    auto drawFrame = [&](double cursor_b, const std::string& note) {
        clearScreen();
        std::cout << "\033[1mPOMDP Belief Simplex\033[0m   "
                  << s0 << " (left)  <-->  " << s1 << " (right)\n";
        std::cout << "Discount γ = " << std::fixed << std::setprecision(2) << ast.discount_factor
                  << "    α-vectors: " << result.alpha_vectors.size()
                  << "    iterations: " << result.iterations << "\n\n";

        // Value curve via a 12-row ASCII band.
        const int H = 12;
        // For each row from top to bottom, threshold = v_max - (row/H)*(v_max - v_min).
        for (int r = 0; r < H; ++r) {
            double thresh = v_max - (static_cast<double>(r) / H) * (v_max - v_min);
            // Print left-edge value label.
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << thresh << " | ";
            for (int i = 0; i < N; ++i) {
                bool above = vs[i] >= thresh;
                if (above) {
                    int col = valueToAnsiColor(vs[i], v_min, v_max);
                    if (use_color)
                        std::cout << "\033[48;5;" << col << "m \033[0m";
                    else
                        std::cout << "#";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "\n";
        }

        // Belief axis with cursor.
        std::cout << "         |";
        int cursor_col = static_cast<int>(cursor_b * (N - 1) + 0.5);
        for (int i = 0; i < N; ++i) {
            if (i == cursor_col) std::cout << (use_color ? "\033[1;37m^\033[0m" : "^");
            else if (i == 0 || i == N-1 || i == N/2) std::cout << "+";
            else std::cout << "-";
        }
        std::cout << "\n";
        std::cout << "         | 0.0 " << std::string(N/2 - 6, ' ')
                  << "b(" << s0 << ")=0.5"
                  << std::string(N - (N/2 - 6) - 14, ' ')
                  << "1.0\n\n";

        // Action region strip.
        std::cout << " action: | ";
        for (int i = 0; i < N; ++i) {
            int col = action_color[acts[i]];
            if (use_color)
                std::cout << "\033[48;5;" << col << "m \033[0m";
            else
                std::cout << acts[i][0];  // first letter of action
        }
        std::cout << "\n";

        // Action legend.
        std::cout << "         legend: ";
        for (const auto& [a, col] : action_color) {
            if (use_color)
                std::cout << "\033[48;5;" << col << "m  \033[0m " << a << "   ";
            else
                std::cout << "[" << a[0] << "] " << a << "  ";
        }
        std::cout << "\n\n";

        // Current belief + action + value.
        auto [v_here, a_here] = evaluate(cursor_b);
        std::cout << " current belief:  b(" << s0 << ") = "
                  << std::fixed << std::setprecision(3) << cursor_b << "    "
                  << "b(" << s1 << ") = " << (1.0 - cursor_b) << "\n";
        std::cout << "       V*(b)   =  " << std::fixed << std::setprecision(3) << v_here << "\n";
        std::cout << "       π*(b)   =  " << a_here << "\n";

        if (!note.empty()) std::cout << "\n " << note << "\n";
    };

    // Initial frame.
    drawFrame(belief[s0], "Initial belief.");
    sleepMs(900);

    // Build a transition index for the belief-update helper.
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> tidx;
    for (const auto& t : ast.transitions)
        tidx[t.source_state + "|" + t.action].emplace_back(t.dest_state, t.probability);

    // Step a few rounds: act under π*(b), get a deterministic observation
    // (the one with highest P(o|b,a)), update belief, repeat.
    std::vector<std::string> obs_list(ast.observations.begin(), ast.observations.end());
    std::sort(obs_list.begin(), obs_list.end());

    for (int step = 1; step <= 8; ++step) {
        double b = belief[s0];
        auto [vv, action] = evaluate(b);

        // Pick the most-likely observation under (b, action).
        std::string best_obs;
        double best_p = -1.0;
        for (const auto& o : obs_list) {
            double p_o = 0.0;
            for (const auto& s_prime : ast.states) {
                double obs_p = 0.0;
                auto it1 = ast.observe_probs.find(s_prime);
                if (it1 != ast.observe_probs.end()) {
                    auto it2 = it1->second.find(action);
                    if (it2 != it1->second.end()) {
                        auto it3 = it2->second.find(o);
                        if (it3 != it2->second.end()) obs_p = it3->second;
                    }
                }
                double inner = 0.0;
                for (const auto& [s, bs] : belief) {
                    auto it = tidx.find(s + "|" + action);
                    if (it == tidx.end()) continue;
                    for (const auto& [sp, tp] : it->second)
                        if (sp == s_prime) inner += bs * tp;
                }
                p_o += obs_p * inner;
            }
            if (p_o > best_p) { best_p = p_o; best_obs = o; }
        }

        if (best_obs.empty() || best_p <= 0.0) {
            drawFrame(belief[s0], "No further useful observation; stopping.");
            sleepMs(900);
            break;
        }

        auto upd = updateBelief(ast, belief, action, best_obs, &tidx);
        if (upd.zero_probability_observation) {
            drawFrame(belief[s0], "Zero-probability observation; uniform fallback.");
            sleepMs(900);
        }
        belief = upd.belief;

        std::ostringstream note;
        note << "step " << step << ":  action = " << action
             << "    observation = " << best_obs
             << "  (P=" << std::fixed << std::setprecision(2) << best_p << ")";
        drawFrame(belief[s0], note.str());
        sleepMs(800);

        // Stop early if a "decisive" action is taken (e.g., OpenLeft/OpenRight).
        // We detect this by looking at whether all observation outcomes from
        // this action are uniform (no information gain) — in Tiger, that's
        // exactly the open-door actions.
        bool informative = false;
        for (const auto& o : obs_list) {
            double p_s0 = 0.0, p_s1 = 0.0;
            auto it1a = ast.observe_probs.find(s0);
            if (it1a != ast.observe_probs.end()) {
                auto it2 = it1a->second.find(action);
                if (it2 != it1a->second.end()) {
                    auto it3 = it2->second.find(o);
                    if (it3 != it2->second.end()) p_s0 = it3->second;
                }
            }
            auto it1b = ast.observe_probs.find(s1);
            if (it1b != ast.observe_probs.end()) {
                auto it2 = it1b->second.find(action);
                if (it2 != it1b->second.end()) {
                    auto it3 = it2->second.find(o);
                    if (it3 != it2->second.end()) p_s1 = it3->second;
                }
            }
            if (std::abs(p_s0 - p_s1) > 1e-6) { informative = true; break; }
        }
        if (!informative) {
            sleepMs(600);
            drawFrame(belief[s0], "Decisive action taken (observation carries no info).");
            sleepMs(900);
            break;
        }
    }

    std::cout << "\n[POMDP animation complete]\n";
}
