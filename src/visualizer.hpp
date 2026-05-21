// ============================================================================
// FILE:    visualizer.hpp
// PURPOSE: Terminal-based ANSI visualization of MDP grid and policy animation.
//          Activated with --animate flag. Pure C++17, no external libraries.
// ============================================================================
#ifndef VISUALIZER_HPP_INCLUDED
#define VISUALIZER_HPP_INCLUDED

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

// Forward declarations — the actual structs are in mdp_compiler.cpp
struct MDP_AST;
struct SolverResult;

// ---- GridInfo: detected grid layout for RrCc-named states ----
struct GridInfo {
    bool is_grid;               // True if all states match RrCc pattern
    int rows, cols;             // Grid dimensions
    std::string goal_state;     // State with highest reward (bright green)
    std::string trap_state;     // State with lowest reward (bright red)
    std::unordered_map<std::string, std::pair<int,int>> state_to_rc; // name -> (row, col)
    std::vector<std::vector<std::string>> grid;  // grid[row][col] = state name or ""
};

// Detect whether the MDP uses a grid layout (RrCc naming)
GridInfo detectGrid(const MDP_AST& ast);

// Check if terminal supports ANSI colors
bool terminalSupportsColor();

// Convert a V* value to a 256-color ANSI background code (red->yellow->green gradient)
int valueToAnsiColor(double value, double v_min, double v_max);

// Get the arrow character for a policy action
std::string actionArrow(const std::string& action);

// Draw the grid with current values and policy, with optional cursor position
void drawGrid(const GridInfo& gi, const MDP_AST& ast,
              const std::unordered_map<std::string, double>& values,
              const std::unordered_map<std::string, std::string>& policy,
              const std::string& cursor_state = "",
              const std::string& stats_text = "",
              bool use_color = true);

// Draw a linear (non-grid) layout for arbitrary MDPs
void drawLinear(const MDP_AST& ast,
                const std::unordered_map<std::string, double>& values,
                const std::unordered_map<std::string, std::string>& policy,
                const std::string& cursor_state = "",
                bool use_color = true);

// Animate Value Iteration: live progress every N iterations
// callback is called with current V values at each display step
void animateValueIteration(const MDP_AST& ast, const SolverResult& final_result,
                           const GridInfo& gi, bool use_color);

// Animate the optimal policy path from a given starting state
void animatePolicyPath(const MDP_AST& ast, const SolverResult& result,
                       const GridInfo& gi, const std::string& start_state,
                       bool use_color);

// Main entry point: runs the full terminal animation sequence
void runTerminalAnimation(const MDP_AST& ast, const SolverResult& result);

// v3.0 — POMDP animation. Pretty-prints the belief simplex (2-state case)
// as an ASCII band of action-region colors, the alpha-vector envelope, and
// then steps through one Listen / observation cycle from the initial belief.
struct PBVIResult;  // forward decl (full def in mdp_compiler.cpp)
void runPOMDPAnimation(const MDP_AST& ast, const PBVIResult& result);

#endif // VISUALIZER_HPP_INCLUDED
