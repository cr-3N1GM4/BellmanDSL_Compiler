// ============================================================================
// FILE:    gridworld_hardcoded.cpp
// PROJECT: MDP-DSL Phase 4 — Hardcoded Comparison
// AUTHOR:  Charvit Rajani (Roll: 240102028)
// DATE:    2026-04-10
//
// PURPOSE: Solve the SAME 4x3 GridWorld as gridworld_4x3.mdp, but using
//          raw C++ arrays and manual index arithmetic — NO DSL.
//
//          This file exists SOLELY to compare against the DSL approach.
//          It demonstrates why general-purpose code is harder to read,
//          more error-prone, and more verbose than the DSL.
//
// COMPILE: g++ -std=c++17 -O2 -Wall -Wextra -o hardcoded gridworld_hardcoded.cpp
// RUN:     ./hardcoded
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Grid dimensions
const int ROWS = 3;
const int COLS = 4;

// Total states = ROWS * COLS = 12, but cell (1,1) is a wall
// We use a flat index: state = row * COLS + col (0-indexed)
// Wall at (row=1, col=1) → index 5

// Actions: 0=Up, 1=Down, 2=Left, 3=Right
const int NUM_ACTIONS = 4;
const int UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3;

// Direction vectors: (delta_row, delta_col) for each action
const int DR[] = {1, -1, 0, 0};   // Up=+1 row, Down=-1 row
const int DC[] = {0, 0, -1, 1};   // Left=-1 col, Right=+1 col

// Parameters
const double GAMMA = 0.9;
const double THETA = 1e-9;
const int MAX_ITER = 10000;

// Goal and Trap positions (row, col), 0-indexed
const int GOAL_R = 2, GOAL_C = 3;  // (row 2, col 3) = top-right
const int TRAP_R = 1, TRAP_C = 3;  // (row 1, col 3) = right of wall
const int WALL_R = 1, WALL_C = 1;  // Wall

// Convert (row, col) to flat index
int toIndex(int r, int c) { return r * COLS + c; }

// Check if (r, c) is valid and not a wall
bool isValid(int r, int c) {
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return false;
    if (r == WALL_R && c == WALL_C) return false;
    return true;
}

// Stochastic movement: intended direction + 2 perpendicular slips
// Returns the result state index after attempting to move in direction 'action' from (r, c)
int moveResult(int r, int c, int action) {
    int nr = r + DR[action];
    int nc = c + DC[action];
    if (!isValid(nr, nc)) return toIndex(r, c);  // Bounce back
    return toIndex(nr, nc);
}

// Get perpendicular directions
int slipLeft(int action) {
    // Up→Left, Down→Right, Left→Down, Right→Up
    const int sl[] = {LEFT, RIGHT, DOWN, UP};
    return sl[action];
}
int slipRight(int action) {
    // Up→Right, Down→Left, Left→Up, Right→Down
    const int sr[] = {RIGHT, LEFT, UP, DOWN};
    return sr[action];
}

int main() {
    const int N = ROWS * COLS;  // 12 cells total

    // Rewards: -0.04 everywhere, +1.0 at goal, -1.0 at trap
    double R[12];
    for (int i = 0; i < N; i++) R[i] = -0.04;
    R[toIndex(GOAL_R, GOAL_C)] = 1.0;
    R[toIndex(TRAP_R, TRAP_C)] = -1.0;

    // Value function: initialise to zero
    double V[12];
    for (int i = 0; i < N; i++) V[i] = 0.0;

    // Which cells are terminal (absorbing)?
    bool terminal[12];
    for (int i = 0; i < N; i++) terminal[i] = false;
    terminal[toIndex(GOAL_R, GOAL_C)] = true;
    terminal[toIndex(TRAP_R, TRAP_C)] = true;

    // Policy: best action at each state
    int policy[12];
    for (int i = 0; i < N; i++) policy[i] = -1;

    // State names for output
    std::string names[12];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            names[toIndex(r, c)] = "R" + std::to_string(r+1) + "C" + std::to_string(c+1);
    std::string actionNames[] = {"Up", "Down", "Left", "Right"};

    std::cout << "=== Hardcoded 4x3 GridWorld Solver ===" << std::endl;
    std::cout << "  gamma = " << GAMMA << ", theta = " << THETA << std::endl;
    std::cout << std::endl;

    // ---- VALUE ITERATION ----
    int iterations = 0;
    for (int iter = 1; iter <= MAX_ITER; iter++) {
        double delta = 0.0;

        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                int s = toIndex(r, c);

                // Skip wall
                if (r == WALL_R && c == WALL_C) continue;

                // Terminal states: V = R/(1-gamma) via iteration
                if (terminal[s]) {
                    double old_val = V[s];
                    V[s] = R[s] + GAMMA * V[s];  // Self-loop
                    double change = std::abs(V[s] - old_val);
                    if (change > delta) delta = change;
                    continue;
                }

                double old_val = V[s];
                double best_q = std::numeric_limits<double>::lowest();

                for (int a = 0; a < NUM_ACTIONS; a++) {
                    // 80% intended, 10% slip left, 10% slip right
                    int s_intended = moveResult(r, c, a);
                    int s_slipL = moveResult(r, c, slipLeft(a));
                    int s_slipR = moveResult(r, c, slipRight(a));

                    double q = 0.8 * V[s_intended]
                             + 0.1 * V[s_slipL]
                             + 0.1 * V[s_slipR];

                    if (q > best_q) {
                        best_q = q;
                        policy[s] = a;
                    }
                }

                V[s] = R[s] + GAMMA * best_q;
                double change = std::abs(V[s] - old_val);
                if (change > delta) delta = change;
            }
        }

        if (delta < THETA) {
            iterations = iter;
            std::cout << "  Converged after " << iter << " iterations." << std::endl;
            std::cout << "  Final delta: " << std::scientific << std::setprecision(2)
                      << delta << std::fixed << std::endl;
            break;
        }
        if (iter == MAX_ITER) {
            iterations = iter;
            std::cout << "  Did NOT converge after " << iter << " iterations." << std::endl;
        }
    }

    // ---- PRINT RESULTS ----
    std::cout << std::endl;
    std::cout << "  " << std::left
              << std::setw(10) << "State"
              << std::setw(10) << "R(s)"
              << std::setw(12) << "V*(s)"
              << "pi*(s)" << std::endl;
    std::cout << "  " << std::string(10+10+12+10, '-') << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    for (int r = ROWS-1; r >= 0; r--) {
        for (int c = 0; c < COLS; c++) {
            int s = toIndex(r, c);
            if (r == WALL_R && c == WALL_C) continue;

            std::string act = terminal[s] ? "(absorb)" : actionNames[policy[s]];
            std::cout << "  " << std::left
                      << std::setw(10) << names[s]
                      << std::setw(10) << R[s]
                      << std::setw(12) << V[s]
                      << act << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "  Lines of code: ~140 (manual index arithmetic, no safety checks)" << std::endl;
    std::cout << "  Compare with:  ~200 lines of .mdp file + existing compiler" << std::endl;

    return 0;
}
