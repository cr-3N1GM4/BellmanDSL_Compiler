// run_all_tests.cpp — Complete MDP-DSL v2.0 test suite
// Compile: g++ -std=c++17 -O2 -DMDP_TESTING_MODE -o run_tests run_all_tests.cpp
// (The -DMDP_TESTING_MODE flag excludes main() from mdp_compiler.cpp)

#define MDP_TESTING_MODE
#include "../src/mdp_compiler.cpp"

// Phase 3B: include mdp_autopsy.cpp (without its main) so its parser and
// solver functions are testable directly.
#define MDP_AUTOPSY_NO_MAIN
#include "../src/mdp_autopsy.cpp"

#include "test_framework.hpp"

#include <sstream>
#include <fstream>

// Helper: create a temporary .mdp file and return its path
std::string writeTempMDP(const std::string& content, const std::string& name = "test") {
    std::string path = "/tmp/mdp_test_" + name + ".mdp";
    std::ofstream f(path);
    f << content;
    f.close();
    return path;
}

// Suppress cout during tests
struct SilentCout {
    std::streambuf* old;
    std::ostringstream sink;
    SilentCout() : old(std::cout.rdbuf(sink.rdbuf())) {}
    ~SilentCout() { std::cout.rdbuf(old); }
};

// ============================================================================
// PARSER TESTS (20 tests)
// ============================================================================

TEST(parse_single_state) {
    SilentCout s;
    auto ast = parseMDPString("STATE: Start\nACTION: Go\nTRANSITION: Start Go Start 1.0\n");
    ASSERT_EQ((int)ast.states.size(), 1);
    ASSERT_TRUE(ast.states.count("Start") > 0);
}

TEST(parse_transition) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nSTATE: B\nACTION: Go\nTRANSITION: A Go B 0.8\n");
    ASSERT_EQ((int)ast.transitions.size(), 1);
    ASSERT_EQ(ast.transitions[0].source_state, std::string("A"));
    ASSERT_EQ(ast.transitions[0].dest_state, std::string("B"));
    ASSERT_NEAR(ast.transitions[0].probability, 0.8, 1e-9);
}

TEST(parse_duplicate_state) {
    SilentCout s;
    auto ast = parseMDPString("STATE: X\nSTATE: X\nACTION: A\nTRANSITION: X A X 1.0\n");
    ASSERT_EQ((int)ast.states.size(), 1);
}

TEST(parse_malformed_transition) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go\n");
    ASSERT_EQ((int)ast.transitions.size(), 0);
}

TEST(parse_discount_clamp_high) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nDISCOUNT: 1.5\n");
    ASSERT_NEAR(ast.discount_factor, 1.0, 1e-9);
}

TEST(parse_discount_clamp_low) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nDISCOUNT: -0.1\n");
    ASSERT_NEAR(ast.discount_factor, 0.0, 1e-9);
}

TEST(parse_case_insensitive) {
    SilentCout s;
    auto ast = parseMDPString("state: X\naction: Go\ntransition: X Go X 1.0\n");
    ASSERT_EQ((int)ast.states.size(), 1);
    ASSERT_EQ((int)ast.actions.size(), 1);
}

TEST(parse_comments_ignored) {
    SilentCout s;
    auto ast = parseMDPString("# comment\nSTATE: A\n# another\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    ASSERT_EQ((int)ast.states.size(), 1);
}

TEST(parse_blank_lines_ignored) {
    SilentCout s;
    auto ast = parseMDPString("\n\nSTATE: A\n\nACTION: Go\n\nTRANSITION: A Go A 1.0\n\n");
    ASSERT_EQ((int)ast.states.size(), 1);
}

TEST(parse_file_not_found) {
    SilentCout s;
    auto ast = parseMDPFile("/nonexistent/path.mdp");
    ASSERT_TRUE(ast.states.empty());
}

TEST(parse_solver_all) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nSOLVER: all\n");
    ASSERT_EQ(ast.solver_mode, std::string("all"));
}

TEST(parse_solver_invalid) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nSOLVER: invalid\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(parse_action_reward) {
    SilentCout s;
    auto ast = parseMDPString("STATE: S\nACTION: Go\nTRANSITION: S Go S 1.0\nACTION_REWARD: S Go 2.5\n");
    ASSERT_NEAR(ast.action_rewards["S"]["Go"], 2.5, 1e-9);
}

TEST(parse_learning_rate) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nLEARNING_RATE: 0.3\n");
    ASSERT_NEAR(ast.learning_rate, 0.3, 1e-9);
}

TEST(parse_episodes) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nEPISODES: 10000\n");
    ASSERT_EQ(ast.episodes, 10000);
}

TEST(parse_random_seed) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nRANDOM_SEED: 123\n");
    ASSERT_EQ(ast.random_seed, 123);
}

TEST(parse_state_type) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: R1C1 GridCell(1,1)\nACTION: Go\nTRANSITION: R1C1 Go R1C1 1.0\n");
    ASSERT_TRUE(ast.state_types.count("GridCell") > 0);
    ASSERT_EQ((int)ast.state_types["GridCell"].fields.size(), 2);
}

TEST(parse_typed_state) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: R1C1 GridCell(1,1)\nACTION: Go\nTRANSITION: R1C1 Go R1C1 1.0\n");
    ASSERT_TRUE(ast.typed_states.count("R1C1") > 0);
    ASSERT_EQ(ast.typed_states["R1C1"].type_name, std::string("GridCell"));
}

TEST(parse_verify) {
    SilentCout s;
    auto ast = parseMDPString("STATE: Goal\nACTION: Stay\nTRANSITION: Goal Stay Goal 1.0\nREWARD: Goal 10.0\nVERIFY: V(Goal) > 9.0\n");
    ASSERT_EQ((int)ast.verify_assertions.size(), 1);
    ASSERT_EQ(ast.verify_assertions[0], std::string("V(Goal) > 9.0"));
}

TEST(parse_grid_preprocessor) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "grid_pp");
    auto ast = parseMDPFile(path);
    ASSERT_TRUE(ast.states.count("R1C1") > 0);
    ASSERT_TRUE(ast.states.count("R2C2") > 0);
    ASSERT_TRUE(ast.transitions.size() > 0);
}

void run_parser_tests() {
    std::cout << "\n--- PARSER TESTS ---\n";
    RUN_TEST(parse_single_state);
    RUN_TEST(parse_transition);
    RUN_TEST(parse_duplicate_state);
    RUN_TEST(parse_malformed_transition);
    RUN_TEST(parse_discount_clamp_high);
    RUN_TEST(parse_discount_clamp_low);
    RUN_TEST(parse_case_insensitive);
    RUN_TEST(parse_comments_ignored);
    RUN_TEST(parse_blank_lines_ignored);
    RUN_TEST(parse_file_not_found);
    RUN_TEST(parse_solver_all);
    RUN_TEST(parse_solver_invalid);
    RUN_TEST(parse_action_reward);
    RUN_TEST(parse_learning_rate);
    RUN_TEST(parse_episodes);
    RUN_TEST(parse_random_seed);
    RUN_TEST(parse_state_type);
    RUN_TEST(parse_typed_state);
    RUN_TEST(parse_verify);
    RUN_TEST(parse_grid_preprocessor);
}

// ============================================================================
// VALIDATOR TESTS (18 tests)
// ============================================================================

TEST(val_source_state_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_source_state_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: B Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
    ASSERT_TRUE(vr.errors[0].find("Undeclared source") != std::string::npos);
}

TEST(val_dest_state_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nSTATE: B\nACTION: Go\nTRANSITION: A Go B 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_dest_state_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go Z 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
    ASSERT_TRUE(vr.errors[0].find("Undeclared destination") != std::string::npos);
}

TEST(val_action_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_action_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Run A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
    ASSERT_TRUE(vr.errors[0].find("Undeclared action") != std::string::npos);
}

TEST(val_prob_sum_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nSTATE: B\nACTION: Go\nTRANSITION: A Go A 0.3\nTRANSITION: A Go B 0.7\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_prob_sum_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nSTATE: B\nACTION: Go\nTRANSITION: A Go A 0.3\nTRANSITION: A Go B 0.67\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
    ASSERT_TRUE(vr.errors[0].find("Probability sum") != std::string::npos);
}

TEST(val_reward_complete_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nREWARD: A 5.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.warnings.empty());
}

TEST(val_reward_complete_warn) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.warnings.size() > 0);
    ASSERT_TRUE(vr.warnings[0].find("no REWARD") != std::string::npos);
}

TEST(val_orphan_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    // A is in transitions, not orphaned
    bool has_orphan = false;
    for (auto& w : vr.warnings) if (w.find("orphaned") != std::string::npos) has_orphan = true;
    ASSERT_FALSE(has_orphan);
}

TEST(val_orphan_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nSTATE: B\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    bool has_orphan = false;
    for (auto& w : vr.warnings) if (w.find("orphaned") != std::string::npos) has_orphan = true;
    ASSERT_TRUE(has_orphan);
}

TEST(val_reward_ref_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nREWARD: A 5.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_reward_ref_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nREWARD: Phantom 99.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(val_action_reward_state_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nACTION_REWARD: A Go 2.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_action_reward_state_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nACTION_REWARD: Z Go 2.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(val_action_reward_action_pass) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nACTION_REWARD: A Go 2.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(val_action_reward_action_fail) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nACTION_REWARD: A Run 2.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

void run_validator_tests() {
    std::cout << "\n--- VALIDATOR TESTS ---\n";
    RUN_TEST(val_source_state_pass); RUN_TEST(val_source_state_fail);
    RUN_TEST(val_dest_state_pass); RUN_TEST(val_dest_state_fail);
    RUN_TEST(val_action_pass); RUN_TEST(val_action_fail);
    RUN_TEST(val_prob_sum_pass); RUN_TEST(val_prob_sum_fail);
    RUN_TEST(val_reward_complete_pass); RUN_TEST(val_reward_complete_warn);
    RUN_TEST(val_orphan_pass); RUN_TEST(val_orphan_fail);
    RUN_TEST(val_reward_ref_pass); RUN_TEST(val_reward_ref_fail);
    RUN_TEST(val_action_reward_state_pass); RUN_TEST(val_action_reward_state_fail);
    RUN_TEST(val_action_reward_action_pass); RUN_TEST(val_action_reward_action_fail);
}

// ============================================================================
// VALUE ITERATION TESTS (10 tests)
// ============================================================================

// Helper: build gridworld_simple AST
MDP_AST buildSimpleGridworld() {
    std::string content = R"(
STATE: Start
STATE: Middle
STATE: Goal
STATE: Trap
ACTION: MoveRight
ACTION: MoveLeft
ACTION: Stay
TRANSITION: Start MoveRight Middle 0.8
TRANSITION: Start MoveRight Start 0.2
TRANSITION: Middle MoveRight Goal 0.7
TRANSITION: Middle MoveRight Trap 0.3
TRANSITION: Middle MoveLeft Start 0.9
TRANSITION: Middle MoveLeft Middle 0.1
TRANSITION: Goal Stay Goal 1.0
TRANSITION: Trap Stay Trap 1.0
REWARD: Start 0.0
REWARD: Middle 1.0
REWARD: Goal 100.0
REWARD: Trap -50.0
DISCOUNT: 0.9
)";
    SilentCout s;
    return parseMDPString(content);
}

TEST(vi_convergence_count) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    auto r = solveValueIteration(ast);
    ASSERT_EQ(r.iterations, 242);
}

TEST(vi_goal_value) {
    SilentCout s;
    auto r = solveValueIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Goal"], 1000.0, 1e-6);
}

TEST(vi_trap_value) {
    SilentCout s;
    auto r = solveValueIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Trap"], -500.0, 1e-6);
}

TEST(vi_middle_value) {
    SilentCout s;
    auto r = solveValueIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Middle"], 496.0, 1e-6);
}

TEST(vi_middle_policy) {
    SilentCout s;
    auto r = solveValueIteration(buildSimpleGridworld());
    ASSERT_EQ(r.policy["Middle"], std::string("MoveRight"));
}

TEST(vi_start_policy) {
    SilentCout s;
    auto r = solveValueIteration(buildSimpleGridworld());
    ASSERT_EQ(r.policy["Start"], std::string("MoveRight"));
}

TEST(vi_4x3_goal) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "vi4x3");
    auto ast = parseMDPFile(path);
    auto r = solveValueIteration(ast);
    ASSERT_NEAR(r.values["R3C4"], 10.0, 1e-6);
}

TEST(vi_4x3_policy_r1c2) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "vi4x3b");
    auto ast = parseMDPFile(path);
    auto r = solveValueIteration(ast);
    ASSERT_EQ(r.policy["R1C2"], std::string("Left"));
}

TEST(vi_4x3_policy_r1c4) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "vi4x3c");
    auto ast = parseMDPFile(path);
    auto r = solveValueIteration(ast);
    ASSERT_EQ(r.policy["R1C4"], std::string("Left"));
}

TEST(vi_gamma1_no_converge) {
    SilentCout s;
    auto ast = parseMDPString("STATE: Only\nACTION: Stay\nTRANSITION: Only Stay Only 1.0\nREWARD: Only 5.0\nDISCOUNT: 1.0\n");
    auto r = solveValueIteration(ast);
    ASSERT_FALSE(r.converged);
}

void run_vi_tests() {
    std::cout << "\n--- VALUE ITERATION TESTS ---\n";
    RUN_TEST(vi_convergence_count);
    RUN_TEST(vi_goal_value);
    RUN_TEST(vi_trap_value);
    RUN_TEST(vi_middle_value);
    RUN_TEST(vi_middle_policy);
    RUN_TEST(vi_start_policy);
    RUN_TEST(vi_4x3_goal);
    RUN_TEST(vi_4x3_policy_r1c2);
    RUN_TEST(vi_4x3_policy_r1c4);
    RUN_TEST(vi_gamma1_no_converge);
}

// ============================================================================
// POLICY ITERATION TESTS (10 tests)
// ============================================================================

TEST(pi_converges_quickly) {
    SilentCout s;
    auto r = solvePolicyIteration(buildSimpleGridworld());
    ASSERT_TRUE(r.policy_improvement_steps <= 10);
}

TEST(pi_goal_value) {
    SilentCout s;
    auto r = solvePolicyIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Goal"], 1000.0, 1e-6);
}

TEST(pi_trap_value) {
    SilentCout s;
    auto r = solvePolicyIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Trap"], -500.0, 1e-6);
}

TEST(pi_middle_value) {
    SilentCout s;
    auto r = solvePolicyIteration(buildSimpleGridworld());
    ASSERT_NEAR(r.values["Middle"], 496.0, 1e-6);
}

TEST(pi_policy_matches_vi) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    auto vi = solveValueIteration(ast);
    auto pi = solvePolicyIteration(ast);
    for (const auto& s : ast.states)
        ASSERT_EQ(vi.policy[s], pi.policy[s]);
}

TEST(pi_4x3_converges) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "pi4x3");
    auto ast = parseMDPFile(path);
    auto r = solvePolicyIteration(ast);
    ASSERT_TRUE(r.policy_improvement_steps <= 15);
}

TEST(pi_4x3_values_match_vi) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "pi4x3v");
    auto ast = parseMDPFile(path);
    auto vi = solveValueIteration(ast);
    auto pi = solvePolicyIteration(ast);
    for (const auto& s : ast.states)
        ASSERT_NEAR(vi.values[s], pi.values[s], 1e-6);
}

TEST(pi_4x3_policy_match) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "pi4x3p");
    auto ast = parseMDPFile(path);
    auto vi = solveValueIteration(ast);
    auto pi = solvePolicyIteration(ast);
    for (const auto& s : ast.states)
        ASSERT_EQ(vi.policy[s], pi.policy[s]);
}

TEST(pi_fewer_iterations_than_vi) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    auto vi = solveValueIteration(ast);
    auto pi = solvePolicyIteration(ast);
    ASSERT_TRUE(pi.policy_improvement_steps < vi.iterations);
}

TEST(pi_bellman_evals_documented) {
    // PI total bellman evals can exceed VI iterations — this is the known tradeoff
    SilentCout s;
    auto ast = buildSimpleGridworld();
    auto pi = solvePolicyIteration(ast);
    ASSERT_TRUE(pi.total_bellman_evaluations > 0);
    // Just document this passes
}

void run_pi_tests() {
    std::cout << "\n--- POLICY ITERATION TESTS ---\n";
    RUN_TEST(pi_converges_quickly);
    RUN_TEST(pi_goal_value);
    RUN_TEST(pi_trap_value);
    RUN_TEST(pi_middle_value);
    RUN_TEST(pi_policy_matches_vi);
    RUN_TEST(pi_4x3_converges);
    RUN_TEST(pi_4x3_values_match_vi);
    RUN_TEST(pi_4x3_policy_match);
    RUN_TEST(pi_fewer_iterations_than_vi);
    RUN_TEST(pi_bellman_evals_documented);
}

// ============================================================================
// Q-LEARNING TESTS (8 tests)
// ============================================================================

TEST(ql_simple_policy_match) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.episodes = 50000; ast.random_seed = 42; ast.learning_rate = 0.1;
    ast.epsilon_start = 0.9; ast.epsilon_decay = 0.995;
    auto vi = solveValueIteration(ast);
    auto ql = solveQLearning(ast);
    int match = 0;
    for (const auto& st : ast.states)
        if (vi.policy.count(st) && ql.policy.count(st) && vi.policy[st] == ql.policy[st]) match++;
    ASSERT_TRUE(match >= 3); // At least 3/4 non-terminal states match
}

TEST(ql_4x3_policy_match) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\nEPISODES: 100000\nRANDOM_SEED: 42\nLEARNING_RATE: 0.1\nEPSILON: 0.9\nEPSILON_DECAY: 0.9995\n", "ql4x3");
    auto ast = parseMDPFile(path);
    auto vi = solveValueIteration(ast);
    auto ql = solveQLearning(ast);
    int match = 0;
    for (const auto& st : ast.states)
        if (vi.policy.count(st) && ql.policy.count(st) && vi.policy[st] == ql.policy[st]) match++;
    ASSERT_TRUE(match >= 6); // At least 6/11 states match (stochastic Q-Learning)
}

TEST(ql_episode_rewards_count) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.episodes = 1000; ast.random_seed = 42;
    auto ql = solveQLearning(ast);
    ASSERT_EQ((int)ql.episode_rewards.size(), 1000);
}

TEST(ql_avg_rewards_count) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.episodes = 1000; ast.random_seed = 42;
    auto ql = solveQLearning(ast);
    ASSERT_EQ((int)ql.avg_rewards_100ep.size(), 10); // 1000/100
}

TEST(ql_lr_zero_error) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nSOLVER: q_learning\nLEARNING_RATE: 0.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(ql_epsilon_decay_1_error) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nSOLVER: q_learning\nEPSILON_DECAY: 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(ql_episodes_zero_error) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nSOLVER: q_learning\nEPISODES: 0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(ql_q_goal_convergence) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.episodes = 50000; ast.random_seed = 42;
    auto ql = solveQLearning(ast);
    double q_goal_stay = ql.Q_table.count("Goal") && ql.Q_table.at("Goal").count("Stay")
                         ? ql.Q_table.at("Goal").at("Stay") : 0.0;
    ASSERT_NEAR(q_goal_stay, 1000.0, 5.0); // Within 5.0 of true V*
}

void run_ql_tests() {
    std::cout << "\n--- Q-LEARNING TESTS ---\n";
    RUN_TEST(ql_simple_policy_match);
    RUN_TEST(ql_4x3_policy_match);
    RUN_TEST(ql_episode_rewards_count);
    RUN_TEST(ql_avg_rewards_count);
    RUN_TEST(ql_lr_zero_error);
    RUN_TEST(ql_epsilon_decay_1_error);
    RUN_TEST(ql_episodes_zero_error);
    RUN_TEST(ql_q_goal_convergence);
}

// ============================================================================
// INTEGRATION TESTS (6 tests)
// ============================================================================

TEST(integ_simple_exits_ok) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
    auto sr = solveValueIteration(ast);
    ASSERT_TRUE(sr.converged);
}

TEST(integ_invalid_blocked) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: B Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(integ_macro_matches_manual) {
    SilentCout s;
    // Macro version
    std::string mpath = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "int_macro");
    auto mast = parseMDPFile(mpath);
    auto mvi = solveValueIteration(mast);

    // Check specific values match known correct results
    ASSERT_NEAR(mvi.values["R3C4"], 10.0, 1e-6);
    ASSERT_NEAR(mvi.values["R2C4"], -10.0, 1e-6);
    ASSERT_NEAR(mvi.values["R1C1"], 4.708, 0.001);
}

TEST(integ_solver_all) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.solver_mode = "all";
    ast.episodes = 1000; ast.random_seed = 42;
    ast.learning_rate = 0.1; ast.epsilon_start = 0.9; ast.epsilon_decay = 0.995;
    auto vi = solveValueIteration(ast);
    auto pi = solvePolicyIteration(ast);
    auto ql = solveQLearning(ast);
    ASSERT_TRUE(vi.converged);
    ASSERT_TRUE(pi.converged);
    ASSERT_TRUE(ql.episode_rewards.size() > 0);
}

TEST(integ_verify_pass) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) > V(Trap)");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 1);
    ASSERT_EQ(vr.fail_count, 0);
}

TEST(integ_verify_fail) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) < 0.0");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.fail_count, 1);
}

void run_integration_tests() {
    std::cout << "\n--- INTEGRATION TESTS ---\n";
    RUN_TEST(integ_simple_exits_ok);
    RUN_TEST(integ_invalid_blocked);
    RUN_TEST(integ_macro_matches_manual);
    RUN_TEST(integ_solver_all);
    RUN_TEST(integ_verify_pass);
    RUN_TEST(integ_verify_fail);
}

// ============================================================================
// TYPE SYSTEM TESTS (8 tests)
// ============================================================================

TEST(type_declaration) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: A GridCell(1,2)\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    ASSERT_TRUE(ast.state_types.count("GridCell") > 0);
    ASSERT_EQ((int)ast.state_types["GridCell"].fields.size(), 2);
}

TEST(type_field_names) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: A GridCell(1,2)\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    ASSERT_EQ(ast.state_types["GridCell"].fields[0].name, std::string("row"));
    ASSERT_EQ(ast.state_types["GridCell"].fields[1].datatype, std::string("int"));
}

TEST(type_valid_annotations) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: R1C1 GridCell(1,1)\nACTION: Go\nTRANSITION: R1C1 Go R1C1 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

TEST(type_undeclared_type_error) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A FakeType(1)\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(type_field_count_mismatch) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: A GridCell(1)\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(type_bad_int_value) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: GridCell row:int col:int\nSTATE: A GridCell(1,hello)\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_FALSE(vr.isValid());
}

TEST(type_untyped_state_ok) {
    SilentCout s;
    auto ast = parseMDPString("STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid()); // No type annotation is fine
}

TEST(type_zero_field) {
    SilentCout s;
    auto ast = parseMDPString("STATE_TYPE: Simple\nSTATE: A Simple\nACTION: Go\nTRANSITION: A Go A 1.0\n");
    auto vr = validateAST(ast);
    ASSERT_TRUE(vr.isValid());
}

void run_type_tests() {
    std::cout << "\n--- TYPE SYSTEM TESTS ---\n";
    RUN_TEST(type_declaration);
    RUN_TEST(type_field_names);
    RUN_TEST(type_valid_annotations);
    RUN_TEST(type_undeclared_type_error);
    RUN_TEST(type_field_count_mismatch);
    RUN_TEST(type_bad_int_value);
    RUN_TEST(type_untyped_state_ok);
    RUN_TEST(type_zero_field);
}

// ============================================================================
// MACRO TESTS (6 tests)
// ============================================================================

TEST(macro_generates_states) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "mg_states");
    auto ast = parseMDPFile(path);
    ASSERT_EQ((int)ast.states.size(), 4); // 2x2 = 4 states, no walls
}

TEST(macro_generates_actions) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "mg_actions");
    auto ast = parseMDPFile(path);
    ASSERT_EQ((int)ast.actions.size(), 4); // Up, Down, Left, Right
}

TEST(macro_wall_exclusion) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 3 4 3 4 2 4\nWALL: 2 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "mg_wall");
    auto ast = parseMDPFile(path);
    ASSERT_EQ((int)ast.states.size(), 11); // 12 - 1 wall
    ASSERT_TRUE(ast.states.count("R2C2") == 0);
}

TEST(macro_absorbing_goal) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\n", "mg_absorb");
    auto ast = parseMDPFile(path);
    // Goal (R2C2) should have self-loops for all 4 actions
    int goal_transitions = 0;
    for (const auto& t : ast.transitions)
        if (t.source_state == "R2C2" && t.dest_state == "R2C2") goal_transitions++;
    ASSERT_EQ(goal_transitions, 4);
}

TEST(macro_correct_discount) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.95\n", "mg_disc");
    auto ast = parseMDPFile(path);
    ASSERT_NEAR(ast.discount_factor, 0.95, 1e-9);
}

TEST(macro_passes_solver_through) {
    SilentCout s;
    std::string path = writeTempMDP("GRID: 2 2 2 2 1 2\nSLIP_MODEL: 0.8 0.1 0.1\nLIVING_REWARD: -0.04\nGOAL_REWARD: 1.0\nTRAP_REWARD: -1.0\nGRID_DISCOUNT: 0.9\nSOLVER: policy_iteration\n", "mg_solver");
    auto ast = parseMDPFile(path);
    ASSERT_EQ(ast.solver_mode, std::string("policy_iteration"));
}

void run_macro_tests() {
    std::cout << "\n--- MACRO TESTS ---\n";
    RUN_TEST(macro_generates_states);
    RUN_TEST(macro_generates_actions);
    RUN_TEST(macro_wall_exclusion);
    RUN_TEST(macro_absorbing_goal);
    RUN_TEST(macro_correct_discount);
    RUN_TEST(macro_passes_solver_through);
}

// ============================================================================
// VERIFY TESTS (8 tests)
// ============================================================================

TEST(verify_v_greater_pass) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) > V(Trap)");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 1);
}

TEST(verify_v_greater_fail) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Trap) > V(Goal)");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.fail_count, 1);
}

TEST(verify_v_literal_pass) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) > 999.0");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 1);
}

TEST(verify_v_literal_fail) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) < 0.0");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.fail_count, 1);
}

TEST(verify_pi_eq_pass) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("pi(Middle) == MoveRight");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 1);
}

TEST(verify_pi_eq_fail) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("pi(Middle) == Stay");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.fail_count, 1);
}

TEST(verify_pi_neq_pass) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("pi(Start) != Stay");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 1);
}

TEST(verify_multiple) {
    SilentCout s;
    auto ast = buildSimpleGridworld();
    ast.verify_assertions.push_back("V(Goal) > 0.0");
    ast.verify_assertions.push_back("V(Trap) < 0.0");
    ast.verify_assertions.push_back("pi(Middle) == MoveRight");
    auto sr = solveValueIteration(ast);
    auto vr = runVerification(sr, ast);
    ASSERT_EQ(vr.pass_count, 3);
    ASSERT_EQ(vr.fail_count, 0);
}

void run_verify_tests() {
    std::cout << "\n--- VERIFY TESTS ---\n";
    RUN_TEST(verify_v_greater_pass);
    RUN_TEST(verify_v_greater_fail);
    RUN_TEST(verify_v_literal_pass);
    RUN_TEST(verify_v_literal_fail);
    RUN_TEST(verify_pi_eq_pass);
    RUN_TEST(verify_pi_eq_fail);
    RUN_TEST(verify_pi_neq_pass);
    RUN_TEST(verify_multiple);
}


// ============================================================================
// v3.0 — POMDP PARSER TESTS (3 tests, ≥6 assertions)
// ============================================================================

TEST(parse_observation_keyword) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: S\nACTION: A\nTRANSITION: S A S 1.0\n"
        "OBSERVATION: o1\nOBSERVATION: o2\n");
    ASSERT_EQ((int)ast.observations.size(), 2);
    ASSERT_TRUE(ast.observations.count("o1") > 0);
    ASSERT_TRUE(ast.observations.count("o2") > 0);
}

TEST(parse_observe_prob_and_initial_belief) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: S1\nSTATE: S2\nACTION: A\n"
        "TRANSITION: S1 A S2 1.0\nTRANSITION: S2 A S2 1.0\n"
        "OBSERVATION: o\n"
        "OBSERVE_PROB: S2 A o 1.0\n"
        "INITIAL_BELIEF: S1 0.3\nINITIAL_BELIEF: S2 0.7\n");
    ASSERT_NEAR(ast.observe_probs["S2"]["A"]["o"], 1.0, 1e-9);
    ASSERT_NEAR(ast.initial_belief["S1"], 0.3, 1e-9);
    ASSERT_NEAR(ast.initial_belief["S2"], 0.7, 1e-9);
}

TEST(parse_belief_state_and_pbvi_knobs) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: X\n"
        "TRANSITION: A X B 1.0\nTRANSITION: B X B 1.0\n"
        "BELIEF_STATE: SureA  A:1.0\n"
        "BELIEF_STATE: Mix    A:0.4  B:0.6\n"
        "ALPHA_VECTORS: 20\nPBVI_ITERATIONS: 50\nHORIZON: 10\n");
    ASSERT_EQ((int)ast.belief_states.size(), 2);
    ASSERT_NEAR(ast.belief_states["Mix"].probs["A"], 0.4, 1e-9);
    ASSERT_NEAR(ast.belief_states["Mix"].probs["B"], 0.6, 1e-9);
    ASSERT_EQ(ast.pbvi_alpha_vectors, 20);
    ASSERT_EQ(ast.pbvi_iterations, 50);
    ASSERT_EQ(ast.horizon, 10);
}

void run_pomdp_parser_tests() {
    std::cout << "\n--- POMDP PARSER TESTS (v3.0) ---\n";
    RUN_TEST(parse_observation_keyword);
    RUN_TEST(parse_observe_prob_and_initial_belief);
    RUN_TEST(parse_belief_state_and_pbvi_knobs);
}


// ============================================================================
// v3.0 — POMDP VALIDATOR TESTS (8 tests covering checks #15–#21)
// ============================================================================

// Build a minimal valid POMDP base; tests will mutate to make it invalid.
static MDP_AST buildMinimalPOMDP() {
    MDP_AST a;
    a.states  = {"S1", "S2"};
    a.actions = {"A"};
    a.transitions.push_back({"S1", "A", "S2", 1.0});
    a.transitions.push_back({"S2", "A", "S2", 1.0});
    a.observations = {"o1", "o2"};
    a.observe_probs["S2"]["A"]["o1"] = 0.7;
    a.observe_probs["S2"]["A"]["o2"] = 0.3;
    a.observe_probs["S1"]["A"]["o1"] = 0.5;
    a.observe_probs["S1"]["A"]["o2"] = 0.5;
    a.rewards["S1"] = 0.0;
    a.rewards["S2"] = 1.0;
    a.discount_factor = 0.9;
    a.solver_mode = "pbvi";
    a.initial_belief["S1"] = 1.0;
    return a;
}

TEST(validator_check15_obs_prob_sum_pass) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    auto r = validateAST(a);
    // Should be valid.
    ASSERT_TRUE(r.isValid());
}

TEST(validator_check15_obs_prob_sum_fail) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    a.observe_probs["S2"]["A"]["o2"] = 0.5;  // sum is now 1.2 -> invalid
    auto r = validateAST(a);
    ASSERT_FALSE(r.isValid());
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 15") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check16_undeclared_obs) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    a.observe_probs["S2"]["A"]["ghost"] = 0.0;  // ghost not in observations set
    // Renormalize to keep check 15 happy.
    a.observe_probs["S2"]["A"]["o1"] = 0.7;
    a.observe_probs["S2"]["A"]["o2"] = 0.3;
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 16") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check17_initial_belief_sum) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    a.initial_belief.clear();
    a.initial_belief["S1"] = 0.4;
    a.initial_belief["S2"] = 0.4;  // sum=0.8 != 1
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 17") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check18_pbvi_requires_belief) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    a.initial_belief.clear();
    a.belief_states.clear();
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 18") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check19_belief_state_bad_ref) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    NamedBelief nb;
    nb.name = "Bad";
    nb.probs["NotAState"] = 1.0;
    a.belief_states["Bad"] = nb;
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 19") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check20_belief_state_bad_sum) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    NamedBelief nb;
    nb.name = "Skewed";
    nb.probs["S1"] = 0.4;
    nb.probs["S2"] = 0.4;  // sum=0.8 != 1
    a.belief_states["Skewed"] = nb;
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 20") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

TEST(validator_check21_horizon_negative) {
    SilentCout s;
    auto a = buildMinimalPOMDP();
    a.horizon = -5;
    auto r = validateAST(a);
    bool found = false;
    for (const auto& e : r.errors) if (e.find("Check 21") != std::string::npos) found = true;
    ASSERT_TRUE(found);
}

void run_pomdp_validator_tests() {
    std::cout << "\n--- POMDP VALIDATOR TESTS (v3.0) ---\n";
    RUN_TEST(validator_check15_obs_prob_sum_pass);
    RUN_TEST(validator_check15_obs_prob_sum_fail);
    RUN_TEST(validator_check16_undeclared_obs);
    RUN_TEST(validator_check17_initial_belief_sum);
    RUN_TEST(validator_check18_pbvi_requires_belief);
    RUN_TEST(validator_check19_belief_state_bad_ref);
    RUN_TEST(validator_check20_belief_state_bad_sum);
    RUN_TEST(validator_check21_horizon_negative);
}


// ============================================================================
// v3.0 — BELIEF UPDATE TESTS (2 tests, ≥4 assertions)
// ============================================================================

// Helper: build the canonical Tiger AST in-code (for direct solver tests).
static MDP_AST buildTigerAST() {
    MDP_AST a;
    a.states  = {"TigerLeft", "TigerRight"};
    a.actions = {"Listen", "OpenLeft", "OpenRight"};

    a.transitions.push_back({"TigerLeft",  "Listen",    "TigerLeft",  1.0});
    a.transitions.push_back({"TigerRight", "Listen",    "TigerRight", 1.0});
    a.transitions.push_back({"TigerLeft",  "OpenLeft",  "TigerLeft",  0.5});
    a.transitions.push_back({"TigerLeft",  "OpenLeft",  "TigerRight", 0.5});
    a.transitions.push_back({"TigerRight", "OpenLeft",  "TigerLeft",  0.5});
    a.transitions.push_back({"TigerRight", "OpenLeft",  "TigerRight", 0.5});
    a.transitions.push_back({"TigerLeft",  "OpenRight", "TigerLeft",  0.5});
    a.transitions.push_back({"TigerLeft",  "OpenRight", "TigerRight", 0.5});
    a.transitions.push_back({"TigerRight", "OpenRight", "TigerLeft",  0.5});
    a.transitions.push_back({"TigerRight", "OpenRight", "TigerRight", 0.5});

    a.observations = {"HearLeft", "HearRight"};

    a.observe_probs["TigerLeft" ]["Listen"]["HearLeft"]  = 0.85;
    a.observe_probs["TigerLeft" ]["Listen"]["HearRight"] = 0.15;
    a.observe_probs["TigerRight"]["Listen"]["HearLeft"]  = 0.15;
    a.observe_probs["TigerRight"]["Listen"]["HearRight"] = 0.85;
    for (const std::string& act : {"OpenLeft", "OpenRight"}) {
        a.observe_probs["TigerLeft" ][act]["HearLeft"]  = 0.5;
        a.observe_probs["TigerLeft" ][act]["HearRight"] = 0.5;
        a.observe_probs["TigerRight"][act]["HearLeft"]  = 0.5;
        a.observe_probs["TigerRight"][act]["HearRight"] = 0.5;
    }

    a.action_rewards["TigerLeft" ]["OpenLeft"]  = -100.0;
    a.action_rewards["TigerRight"]["OpenLeft"]  =   10.0;
    a.action_rewards["TigerLeft" ]["OpenRight"] =   10.0;
    a.action_rewards["TigerRight"]["OpenRight"] = -100.0;
    a.action_rewards["TigerLeft" ]["Listen"]    =   -1.0;
    a.action_rewards["TigerRight"]["Listen"]    =   -1.0;

    a.initial_belief["TigerLeft" ] = 0.5;
    a.initial_belief["TigerRight"] = 0.5;

    NamedBelief uni; uni.name = "Uniform";
    uni.probs["TigerLeft"] = 0.5; uni.probs["TigerRight"] = 0.5;
    a.belief_states["Uniform"] = uni;

    NamedBelief asl; asl.name = "AlmostSureLeft";
    asl.probs["TigerLeft"] = 0.97; asl.probs["TigerRight"] = 0.03;
    a.belief_states["AlmostSureLeft"] = asl;

    a.discount_factor = 0.95;
    a.solver_mode = "pbvi";
    a.pbvi_alpha_vectors = 30;
    a.pbvi_iterations    = 200;
    return a;
}

TEST(belief_update_listen_hearleft_from_uniform) {
    SilentCout s;
    auto ast = buildTigerAST();
    std::unordered_map<std::string, double> b = {{"TigerLeft", 0.5}, {"TigerRight", 0.5}};
    auto out = updateBelief(ast, b, "Listen", "HearLeft");
    // Bayes: P(TL|HearLeft, Listen) = 0.85·0.5 / (0.85·0.5 + 0.15·0.5) = 0.85.
    ASSERT_FALSE(out.zero_probability_observation);
    ASSERT_NEAR(out.belief["TigerLeft"],  0.85, 1e-9);
    ASSERT_NEAR(out.belief["TigerRight"], 0.15, 1e-9);
}

TEST(belief_update_zero_prob_obs_returns_uniform) {
    SilentCout s;
    // POMDP where observation "ghost" has zero probability under every (s,a).
    MDP_AST a;
    a.states  = {"X", "Y"};
    a.actions = {"go"};
    a.transitions.push_back({"X", "go", "Y", 1.0});
    a.transitions.push_back({"Y", "go", "Y", 1.0});
    a.observations = {"ghost", "real"};
    a.observe_probs["X"]["go"]["real"]  = 1.0;
    a.observe_probs["X"]["go"]["ghost"] = 0.0;
    a.observe_probs["Y"]["go"]["real"]  = 1.0;
    a.observe_probs["Y"]["go"]["ghost"] = 0.0;
    a.discount_factor = 0.9;

    std::unordered_map<std::string, double> b = {{"X", 1.0}, {"Y", 0.0}};
    auto out = updateBelief(a, b, "go", "ghost");
    // P(ghost | any b, go) = 0  =>  warn + uniform fallback.
    ASSERT_TRUE(out.zero_probability_observation);
    ASSERT_NEAR(out.belief["X"], 0.5, 1e-9);
    ASSERT_NEAR(out.belief["Y"], 0.5, 1e-9);
}

void run_belief_update_tests() {
    std::cout << "\n--- BELIEF UPDATE TESTS (v3.0) ---\n";
    RUN_TEST(belief_update_listen_hearleft_from_uniform);
    RUN_TEST(belief_update_zero_prob_obs_returns_uniform);
}


// ============================================================================
// v3.0 — PBVI SOLVER TESTS (5 tests, ≥10 assertions including Tiger)
// ============================================================================

TEST(pbvi_returns_alpha_vectors) {
    SilentCout s;
    auto ast = buildTigerAST();
    auto r = solvePBVI(ast);
    // PBVI should retain at least 2 useful alpha vectors. The "do not over-prune"
    // trap from the spec warns that going below ~3 destroys solution quality;
    // we test a lower bound of ≥3 to catch the bug.
    ASSERT_TRUE(r.alpha_vectors.size() >= 3);
    ASSERT_TRUE(r.iterations >= 1);
}

TEST(pbvi_tiger_listen_at_uniform) {
    SilentCout s;
    auto ast = buildTigerAST();
    auto r = solvePBVI(ast);
    // Uniform belief is the canonical "stay & gather info" point.
    ASSERT_TRUE(r.policy_at_belief.count("Uniform") > 0);
    ASSERT_EQ(r.policy_at_belief["Uniform"], std::string("Listen"));
}

TEST(pbvi_tiger_open_at_high_confidence) {
    SilentCout s;
    auto ast = buildTigerAST();
    auto r = solvePBVI(ast);
    // At 0.97 belief that the tiger is on the left, the optimal action is to
    // open the door on the RIGHT (away from the tiger).
    ASSERT_TRUE(r.policy_at_belief.count("AlmostSureLeft") > 0);
    ASSERT_EQ(r.policy_at_belief["AlmostSureLeft"], std::string("OpenRight"));
}

TEST(pbvi_tiger_value_increases_with_certainty) {
    SilentCout s;
    auto ast = buildTigerAST();
    auto r = solvePBVI(ast);
    double v_uni = r.values_at_belief["Uniform"];
    double v_sure = r.values_at_belief["AlmostSureLeft"];
    // Higher certainty must yield strictly higher value (we can act!).
    ASSERT_TRUE(v_sure > v_uni);
    // Both should be finite reals.
    ASSERT_TRUE(std::isfinite(v_uni));
    ASSERT_TRUE(std::isfinite(v_sure));
}

TEST(pbvi_symmetric_almost_sure_right) {
    SilentCout s;
    auto ast = buildTigerAST();
    NamedBelief asr; asr.name = "AlmostSureRight";
    asr.probs["TigerLeft"] = 0.03; asr.probs["TigerRight"] = 0.97;
    ast.belief_states["AlmostSureRight"] = asr;
    auto r = solvePBVI(ast);
    // By symmetry the optimal action must be OpenLeft.
    ASSERT_TRUE(r.policy_at_belief.count("AlmostSureRight") > 0);
    ASSERT_EQ(r.policy_at_belief["AlmostSureRight"], std::string("OpenLeft"));
}

void run_pbvi_tests() {
    std::cout << "\n--- PBVI SOLVER TESTS (v3.0) ---\n";
    RUN_TEST(pbvi_returns_alpha_vectors);
    RUN_TEST(pbvi_tiger_listen_at_uniform);
    RUN_TEST(pbvi_tiger_open_at_high_confidence);
    RUN_TEST(pbvi_tiger_value_increases_with_certainty);
    RUN_TEST(pbvi_symmetric_almost_sure_right);
}


// ============================================================================
// v3.0 — INTEGRATION TESTS (Tiger end-to-end + verifier + unicode π)
// ============================================================================

TEST(tiger_integration_verify_assertions_pass) {
    SilentCout s;
    auto ast = buildTigerAST();
    ast.verify_assertions.push_back("pi(Uniform) == Listen");
    ast.verify_assertions.push_back("pi(AlmostSureLeft) == OpenRight");

    auto r = solvePBVI(ast);
    // Merge belief-space values/policies into the SolverResult view that the
    // verifier consumes — exactly the same wiring `main` does.
    SolverResult vsr;
    for (const auto& [k, v] : r.values_at_belief)  vsr.values[k] = v;
    for (const auto& [k, a] : r.policy_at_belief)  vsr.policy[k] = a;
    auto vr = runVerification(vsr, ast);
    ASSERT_EQ(vr.pass_count, 2);
    ASSERT_EQ(vr.fail_count, 0);
}

TEST(verifier_accepts_unicode_pi) {
    SilentCout s;
    auto ast = buildTigerAST();
    // Unicode π (UTF-8: CF 80) should be normalized to "pi" by the verifier.
    ast.verify_assertions.push_back("\xCF\x80(Uniform) == Listen");

    auto r = solvePBVI(ast);
    SolverResult vsr;
    for (const auto& [k, v] : r.values_at_belief) vsr.values[k] = v;
    for (const auto& [k, a] : r.policy_at_belief) vsr.policy[k] = a;
    auto vr = runVerification(vsr, ast);
    ASSERT_EQ(vr.pass_count, 1);
    ASSERT_EQ(vr.fail_count, 0);
}

TEST(backward_compat_v2_no_pomdp_fields_untouched) {
    SilentCout s;
    // A fresh v2.0-style MDP must produce empty POMDP fields in the AST.
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: go\n"
        "TRANSITION: A go B 1.0\nTRANSITION: B go B 1.0\n"
        "REWARD: B 1.0\nDISCOUNT: 0.9\n");
    ASSERT_EQ((int)ast.observations.size(), 0);
    ASSERT_EQ((int)ast.observe_probs.size(), 0);
    ASSERT_EQ((int)ast.initial_belief.size(), 0);
    ASSERT_EQ((int)ast.belief_states.size(), 0);
    // VI must still work with empty POMDP fields.
    auto v = solveValueIteration(ast);
    ASSERT_TRUE(v.converged);
}

void run_pomdp_integration_tests() {
    std::cout << "\n--- POMDP INTEGRATION TESTS (v3.0) ---\n";
    RUN_TEST(tiger_integration_verify_assertions_pass);
    RUN_TEST(verifier_accepts_unicode_pi);
    RUN_TEST(backward_compat_v2_no_pomdp_fields_untouched);
}


// ============================================================================
// v3.0 — HMM-MDP BRIDGE TESTS (Phase 2A)
// ============================================================================

// Generate a synthetic 2-state Gaussian HMM CSV with KNOWN parameters.
// State 0: mu=-1.0, sd=0.4    State 1: mu=+1.0, sd=0.4
// Diagonal-favouring transition (0.9 self-stay).
// Returns the temporary file path; caller is responsible for not depending
// on it staying around between processes (we recreate it in each test).
static std::string writeSyntheticHmmCsv(const std::string& path,
                                        int T = 800, int seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> g0(-1.0, 0.4);
    std::normal_distribution<double> g1( 1.0, 0.4);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int state = 0;
    std::ofstream f(path);
    for (int t = 0; t < T; ++t) {
        double x = (state == 0) ? g0(rng) : g1(rng);
        f << std::fixed << std::setprecision(6) << x << "\n";
        // Self-stay with prob 0.9.
        if (u(rng) > 0.9) state = 1 - state;
    }
    return path;
}

TEST(hmm_parse_keywords) {
    SilentCout s;
    auto ast = parseMDPString(
        "HMM_STATES: Low High\n"
        "EMISSION_TYPE: gaussian\n"
        "FIT_HMM: /tmp/__nonexistent__.csv iterations=50\n"
        "HMM_SEED: 7\n"
        "BRIDGE: hmm -> mdp\n"
        "ACTION: Hold\n");
    ASSERT_EQ((int)ast.hmm_states.size(), 2);
    ASSERT_EQ(ast.hmm_states[0], std::string("Low"));
    ASSERT_EQ(ast.hmm_states[1], std::string("High"));
    ASSERT_EQ(ast.hmm_emission_type, std::string("gaussian"));
    ASSERT_EQ(ast.hmm_fit_iterations, 50);
    ASSERT_EQ(ast.hmm_seed, 7);
    ASSERT_TRUE(ast.hmm_bridge_enabled);
}

TEST(hmm_validator_22_bridge_needs_states) {
    SilentCout s;
    auto ast = parseMDPString(
        "ACTION: A\n"
        "STATE: X\nTRANSITION: X A X 1.0\n"
        "FIT_HMM: /tmp/some.csv\n"
        "BRIDGE: hmm -> mdp\n");
    auto r = validateAST(ast);
    bool found22 = false;
    for (const auto& e : r.errors) if (e.find("Check 22") != std::string::npos) found22 = true;
    ASSERT_TRUE(found22);
}

TEST(hmm_validator_23_bridge_needs_csv) {
    SilentCout s;
    auto ast = parseMDPString(
        "ACTION: A\nSTATE: X\nTRANSITION: X A X 1.0\n"
        "HMM_STATES: Low High\n"
        "BRIDGE: hmm -> mdp\n");
    auto r = validateAST(ast);
    bool found23 = false;
    for (const auto& e : r.errors) if (e.find("Check 23") != std::string::npos) found23 = true;
    ASSERT_TRUE(found23);
}

TEST(hmm_validator_24_bad_emission_type) {
    SilentCout s;
    auto ast = parseMDPString(
        "ACTION: A\nSTATE: X\nTRANSITION: X A X 1.0\n"
        "EMISSION_TYPE: poisson\n");
    auto r = validateAST(ast);
    bool found24 = false;
    for (const auto& e : r.errors) if (e.find("Check 24") != std::string::npos) found24 = true;
    ASSERT_TRUE(found24);
}

TEST(hmm_validator_25_negative_seed) {
    SilentCout s;
    auto ast = parseMDPString(
        "ACTION: A\nSTATE: X\nTRANSITION: X A X 1.0\n"
        "HMM_SEED: -1\n");
    auto r = validateAST(ast);
    bool found25 = false;
    for (const auto& e : r.errors) if (e.find("Check 25") != std::string::npos) found25 = true;
    ASSERT_TRUE(found25);
}

TEST(hmm_fit_recovers_known_means) {
    SilentCout s;
    std::string csv = writeSyntheticHmmCsv("/tmp/__hmm_test_means__.csv", 800, 42);
    std::vector<double> obs;
    std::ifstream f(csv);
    double x;
    while (f >> x) obs.push_back(x);
    ASSERT_TRUE((int)obs.size() >= 700);
    HMMFitResult r = fitGaussianHMM(obs, 2, 200, 42);
    // Means should be near +/- 1.0; HMM may swap state labels (label switching
    // is the standard ambiguity), so we test the SET, not specific indices.
    double m_lo = std::min(r.mu[0], r.mu[1]);
    double m_hi = std::max(r.mu[0], r.mu[1]);
    ASSERT_TRUE(m_lo < -0.7 && m_lo > -1.3);
    ASSERT_TRUE(m_hi >  0.7 && m_hi <  1.3);
}

TEST(hmm_fit_recovers_known_stds) {
    SilentCout s;
    std::string csv = writeSyntheticHmmCsv("/tmp/__hmm_test_stds__.csv", 800, 42);
    std::vector<double> obs;
    std::ifstream f(csv); double x;
    while (f >> x) obs.push_back(x);
    HMMFitResult r = fitGaussianHMM(obs, 2, 200, 42);
    // Both stds should be near 0.4.
    ASSERT_TRUE(r.sd[0] > 0.25 && r.sd[0] < 0.55);
    ASSERT_TRUE(r.sd[1] > 0.25 && r.sd[1] < 0.55);
}

TEST(hmm_fit_log_likelihood_increases) {
    SilentCout s;
    std::string csv = writeSyntheticHmmCsv("/tmp/__hmm_test_ll__.csv", 600, 7);
    std::vector<double> obs;
    std::ifstream f(csv); double x;
    while (f >> x) obs.push_back(x);
    HMMFitResult r1 = fitGaussianHMM(obs, 2, 5,   42);
    HMMFitResult r2 = fitGaussianHMM(obs, 2, 100, 42);
    // More iterations should not decrease log-likelihood (EM monotonicity).
    ASSERT_TRUE(r2.log_likelihood >= r1.log_likelihood - 1e-6);
    ASSERT_TRUE(std::isfinite(r2.log_likelihood));
}

TEST(hmm_fit_handles_long_sequence_without_underflow) {
    SilentCout s;
    // Sequence length 2000: with naive (non-scaled) forward-backward this
    // collapses to zero. The scaled algorithm must still produce a finite
    // log-likelihood.
    std::string csv = writeSyntheticHmmCsv("/tmp/__hmm_test_long__.csv", 2000, 11);
    std::vector<double> obs;
    std::ifstream f(csv); double x;
    while (f >> x) obs.push_back(x);
    HMMFitResult r = fitGaussianHMM(obs, 2, 50, 42);
    ASSERT_TRUE(std::isfinite(r.log_likelihood));
    ASSERT_TRUE(r.log_likelihood > -1.0e9);
}

TEST(hmm_fit_diagonal_transition_recovered) {
    SilentCout s;
    std::string csv = writeSyntheticHmmCsv("/tmp/__hmm_test_diag__.csv", 1000, 99);
    std::vector<double> obs;
    std::ifstream f(csv); double x;
    while (f >> x) obs.push_back(x);
    HMMFitResult r = fitGaussianHMM(obs, 2, 200, 42);
    // Diagonal mass should dominate (true diag = 0.9).
    double diag_avg = 0.5 * (r.A[0][0] + r.A[1][1]);
    ASSERT_TRUE(diag_avg > 0.75 && diag_avg < 0.99);
}

TEST(hmm_bridge_populates_transitions) {
    SilentCout s;
    // Tiny CSV with two clusters (5 negatives, 5 positives).
    std::ofstream f("/tmp/__hmm_bridge_test__.csv");
    for (int i = 0; i < 50; ++i) f << "-1.0\n";
    for (int i = 0; i < 50; ++i) f << "1.0\n";
    f.close();

    auto ast = parseMDPString(
        "HMM_STATES: A B\n"
        "EMISSION_TYPE: gaussian\n"
        "FIT_HMM: /tmp/__hmm_bridge_test__.csv iterations=50\n"
        "HMM_SEED: 42\n"
        "ACTION: Hold\n"
        "BRIDGE: hmm -> mdp\n"
        "ACTION_REWARD: A Hold 0.0\n"
        "ACTION_REWARD: B Hold 1.0\n"
        "DISCOUNT: 0.9\n"
        "SOLVER: value_iteration\n");
    auto errors = bridgeHMMToMDP(ast);
    ASSERT_TRUE(errors.empty());
    ASSERT_TRUE(ast.hmm_fitted);
    // 1 action * 2 source states * 2 dest states = 4 transitions added.
    ASSERT_TRUE((int)ast.transitions.size() >= 2);
    // Both A and B should now be registered as MDP states.
    ASSERT_TRUE(ast.states.count("A") > 0);
    ASSERT_TRUE(ast.states.count("B") > 0);
}

TEST(hmm_bridge_does_not_override_explicit_transitions) {
    SilentCout s;
    std::ofstream f("/tmp/__hmm_override_test__.csv");
    for (int i = 0; i < 50; ++i) f << "-1.0\n";
    for (int i = 0; i < 50; ++i) f << "1.0\n";
    f.close();

    auto ast = parseMDPString(
        "HMM_STATES: A B\n"
        "EMISSION_TYPE: gaussian\n"
        "FIT_HMM: /tmp/__hmm_override_test__.csv iterations=50\n"
        "ACTION: Hold\n"
        "TRANSITION: A Hold A 1.0\n"      // explicit — should be preserved
        "BRIDGE: hmm -> mdp\n");
    auto errors = bridgeHMMToMDP(ast);
    ASSERT_TRUE(errors.empty());
    // The explicit (A, Hold, A, 1.0) row must survive.
    bool found = false;
    for (const auto& t : ast.transitions) {
        if (t.source_state == "A" && t.action == "Hold" && t.dest_state == "A"
            && std::abs(t.probability - 1.0) < 1e-9) {
            found = true; break;
        }
    }
    ASSERT_TRUE(found);
}

TEST(hmm_emission_type_case_insensitive) {
    SilentCout s;
    auto a = parseMDPString("ACTION: A\nSTATE: X\nTRANSITION: X A X 1.0\n"
                            "EMISSION_TYPE: Gaussian\n");
    ASSERT_EQ(a.hmm_emission_type, std::string("gaussian"));
}

TEST(hmm_csv_parser_skips_header_and_blanks) {
    SilentCout s;
    {
        std::ofstream f("/tmp/__hmm_csv_test__.csv");
        f << "date,return\n";
        f << "\n";
        f << "# comment line\n";
        f << "2024-01-01,0.5\n";
        f << "2024-01-02,-0.5\n";
        f << "2024-01-03,1.5\n";
    }
    std::string err;
    auto data = readScalarCSV("/tmp/__hmm_csv_test__.csv", err);
    ASSERT_TRUE(err.empty());
    ASSERT_EQ((int)data.size(), 3);
    ASSERT_NEAR(data[0],  0.5, 1e-9);
    ASSERT_NEAR(data[1], -0.5, 1e-9);
    ASSERT_NEAR(data[2],  1.5, 1e-9);
}

TEST(hmm_csv_parser_reports_missing_file) {
    SilentCout s;
    std::string err;
    auto data = readScalarCSV("/tmp/__never_exists__.csv", err);
    ASSERT_FALSE(err.empty());
    ASSERT_EQ((int)data.size(), 0);
}

void run_hmm_tests() {
    std::cout << "\n--- HMM / BAUM-WELCH TESTS (v3.0 Phase 2A) ---\n";
    RUN_TEST(hmm_parse_keywords);
    RUN_TEST(hmm_validator_22_bridge_needs_states);
    RUN_TEST(hmm_validator_23_bridge_needs_csv);
    RUN_TEST(hmm_validator_24_bad_emission_type);
    RUN_TEST(hmm_validator_25_negative_seed);
    RUN_TEST(hmm_fit_recovers_known_means);
    RUN_TEST(hmm_fit_recovers_known_stds);
    RUN_TEST(hmm_fit_log_likelihood_increases);
    RUN_TEST(hmm_fit_handles_long_sequence_without_underflow);
    RUN_TEST(hmm_fit_diagonal_transition_recovered);
    RUN_TEST(hmm_bridge_populates_transitions);
    RUN_TEST(hmm_bridge_does_not_override_explicit_transitions);
    RUN_TEST(hmm_emission_type_case_insensitive);
    RUN_TEST(hmm_csv_parser_skips_header_and_blanks);
    RUN_TEST(hmm_csv_parser_reports_missing_file);
}


// ============================================================================
// v3.0 PHASE 3A — REWARD AUTOPSY ENGINE TESTS
// ============================================================================
// Each test crafts a small pathological MDP and asserts that the matching
// autopsy pass fires (or doesn't fire when the MDP is clean). Tests cover:
//   - All 6 failure classes (positive + negative for each)
//   - Severity escalation (WARN vs ERROR)
//   - REPAIR command (already-satisfied, success, no-fix)
//   - Assertion parser
//   - Exit-code semantics (via the AutopsyReport API)
//   - Idempotency (running --diagnose twice = same findings)
// ============================================================================

// -------- Class 4: Magnitude Imbalance --------
TEST(autopsy_magnitude_error_below_0p1pct) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 1000.0\n"
        "REWARD: A 0.001\n"           // 0.0001% of total -> ERROR
        "DISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "reward");
    bool found_err = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::MagnitudeImbalance &&
            f.severity == FindingSeverity::ERROR_LEVEL) found_err = true;
    ASSERT_TRUE(found_err);
}

TEST(autopsy_magnitude_warn_between_0p1_and_1pct) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 100.0\n"
        "REWARD: A 0.5\n"             // ~0.5% -> WARN
        "DISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "reward");
    bool found_warn = false, no_err = true;
    for (const auto& f : report.findings) {
        if (f.cls == FindingClass::MagnitudeImbalance) {
            if (f.severity == FindingSeverity::WARN)        found_warn = true;
            if (f.severity == FindingSeverity::ERROR_LEVEL) no_err = false;
        }
    }
    ASSERT_TRUE(found_warn);
    ASSERT_TRUE(no_err);
}

TEST(autopsy_magnitude_clean_no_findings) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 10.0\nREWARD: A 5.0\n"
        "DISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "reward");
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::MagnitudeImbalance);
}

// -------- Class 1: Reward Myopia --------
TEST(autopsy_myopia_fires_long_corridor_low_gamma) {
    SilentCout s;
    // 6-state corridor, gamma=0.2 (eff horizon ≈ 4.3, goal 5 steps away).
    auto ast = parseMDPString(
        "STATE: S0\nSTATE: S1\nSTATE: S2\nSTATE: S3\nSTATE: S4\nSTATE: Goal\n"
        "ACTION: R\n"
        "TRANSITION: S0 R S1 1.0\nTRANSITION: S1 R S2 1.0\n"
        "TRANSITION: S2 R S3 1.0\nTRANSITION: S3 R S4 1.0\n"
        "TRANSITION: S4 R Goal 1.0\nTRANSITION: Goal R Goal 1.0\n"
        "REWARD: Goal 100.0\n"
        "REWARD: S0 -1.0\nREWARD: S1 -1.0\nREWARD: S2 -1.0\n"
        "REWARD: S3 -1.0\nREWARD: S4 -1.0\n"
        "DISCOUNT: 0.2\n");
    auto report = runAutopsy(ast, "reward");
    bool found = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::RewardMyopia &&
            f.severity == FindingSeverity::ERROR_LEVEL) found = true;
    ASSERT_TRUE(found);
}

TEST(autopsy_myopia_does_not_fire_high_gamma) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: S0\nSTATE: S1\nSTATE: Goal\n"
        "ACTION: R\n"
        "TRANSITION: S0 R S1 1.0\nTRANSITION: S1 R Goal 1.0\nTRANSITION: Goal R Goal 1.0\n"
        "REWARD: Goal 100.0\nREWARD: S0 -1.0\nREWARD: S1 -1.0\n"
        "DISCOUNT: 0.95\n");
    auto report = runAutopsy(ast, "reward");
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::RewardMyopia);
}

TEST(autopsy_myopia_suggests_gamma_above_current) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: S0\nSTATE: S1\nSTATE: S2\nSTATE: S3\nSTATE: Goal\n"
        "ACTION: R\n"
        "TRANSITION: S0 R S1 1.0\nTRANSITION: S1 R S2 1.0\nTRANSITION: S2 R S3 1.0\n"
        "TRANSITION: S3 R Goal 1.0\nTRANSITION: Goal R Goal 1.0\n"
        "REWARD: Goal 100.0\nREWARD: S0 -1.0\nREWARD: S1 -1.0\nREWARD: S2 -1.0\nREWARD: S3 -1.0\n"
        "DISCOUNT: 0.2\n");
    auto report = runAutopsy(ast, "reward");
    for (const auto& f : report.findings) {
        if (f.cls == FindingClass::RewardMyopia) {
            auto it = f.numeric_fields.find("suggested_gamma_min");
            ASSERT_TRUE(it != f.numeric_fields.end());
            ASSERT_TRUE(it->second > 0.2);
            ASSERT_TRUE(it->second <= 0.999);
        }
    }
}

// -------- Class 5: Dead State --------
TEST(autopsy_dead_state_detects_isolated_pair) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: Start\nSTATE: Goal\nSTATE: Trap1\nSTATE: Trap2\n"
        "ACTION: F\n"
        "TRANSITION: Start F Goal 1.0\nTRANSITION: Goal F Goal 1.0\n"
        "TRANSITION: Trap1 F Trap2 1.0\nTRANSITION: Trap2 F Trap1 1.0\n"
        "REWARD: Goal 10.0\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "structural");
    bool found = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::DeadState && f.numeric_fields.at("dead_state_count") == 2)
            found = true;
    ASSERT_TRUE(found);
}

TEST(autopsy_dead_state_does_not_fire_clean) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: G\n"
        "TRANSITION: A G B 1.0\nTRANSITION: B G B 1.0\n"
        "REWARD: B 1.0\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "structural");
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::DeadState);
}

// -------- Class 3: Discount Cliff --------
TEST(autopsy_discount_cliff_fires_when_policy_flips) {
    SilentCout s;
    // Right pays +10 (3 hops) but costs -2 per step; Left is free-but-useless.
    // At low gamma: Left wins (cost dominates); high gamma: Right wins.
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: Goal\n"
        "ACTION: Right\nACTION: Left\n"
        "TRANSITION: A Right B 1.0\nTRANSITION: A Left A 1.0\n"
        "TRANSITION: B Right Goal 1.0\nTRANSITION: B Left A 1.0\n"
        "TRANSITION: Goal Right Goal 1.0\nTRANSITION: Goal Left Goal 1.0\n"
        "REWARD: Goal 10.0\n"
        "ACTION_REWARD: A Right -2.0\nACTION_REWARD: B Right -2.0\n"
        "ACTION_REWARD: A Left 0.0\nACTION_REWARD: B Left 0.0\n"
        "DISCOUNT: 0.5\n");
    auto report = runAutopsy(ast, "solver");
    bool found = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::DiscountCliff) found = true;
    ASSERT_TRUE(found);
}

TEST(autopsy_discount_cliff_records_gamma_and_state) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: Goal\n"
        "ACTION: Right\nACTION: Left\n"
        "TRANSITION: A Right B 1.0\nTRANSITION: A Left A 1.0\n"
        "TRANSITION: B Right Goal 1.0\nTRANSITION: B Left A 1.0\n"
        "TRANSITION: Goal Right Goal 1.0\nTRANSITION: Goal Left Goal 1.0\n"
        "REWARD: Goal 10.0\n"
        "ACTION_REWARD: A Right -2.0\nACTION_REWARD: B Right -2.0\n"
        "DISCOUNT: 0.5\n");
    auto report = runAutopsy(ast, "solver");
    for (const auto& f : report.findings) {
        if (f.cls == FindingClass::DiscountCliff) {
            ASSERT_TRUE(f.numeric_fields.count("cliff_gamma") > 0);
            ASSERT_TRUE(f.numeric_fields.count("distance")    > 0);
            ASSERT_TRUE(!f.text_fields.at("state").empty());
        }
    }
}

// -------- Class 2: Reward Hacking --------
TEST(autopsy_reward_hacking_detects_attractive_cycle) {
    SilentCout s;
    // Cycle S2<->S3 with +12 per step beats the goal at +50 one-shot.
    auto ast = parseMDPString(
        "STATE: Start\nSTATE: S2\nSTATE: S3\nSTATE: S4\nSTATE: Goal\n"
        "ACTION: GoToCycle\nACTION: TraverseCycle\nACTION: GoToGoal\nACTION: Stay\n"
        "TRANSITION: Start GoToCycle S2 1.0\n"
        "TRANSITION: Start GoToGoal S4 1.0\n"
        "TRANSITION: Start Stay Start 1.0\n"
        "TRANSITION: Start TraverseCycle Start 1.0\n"
        "TRANSITION: S2 TraverseCycle S3 1.0\nTRANSITION: S2 GoToCycle S2 1.0\n"
        "TRANSITION: S2 GoToGoal S2 1.0\nTRANSITION: S2 Stay S2 1.0\n"
        "TRANSITION: S3 TraverseCycle S2 1.0\nTRANSITION: S3 GoToCycle S3 1.0\n"
        "TRANSITION: S3 GoToGoal S3 1.0\nTRANSITION: S3 Stay S3 1.0\n"
        "TRANSITION: S4 GoToGoal Goal 1.0\nTRANSITION: S4 TraverseCycle S4 1.0\n"
        "TRANSITION: S4 GoToCycle S4 1.0\nTRANSITION: S4 Stay S4 1.0\n"
        "TRANSITION: Goal Stay Goal 1.0\nTRANSITION: Goal TraverseCycle Goal 1.0\n"
        "TRANSITION: Goal GoToCycle Goal 1.0\nTRANSITION: Goal GoToGoal Goal 1.0\n"
        "ACTION_REWARD: S2 TraverseCycle 12.0\n"
        "ACTION_REWARD: S3 TraverseCycle 12.0\n"
        "ACTION_REWARD: S4 GoToGoal 50.0\n"
        "DISCOUNT: 0.95\n");
    auto report = runAutopsy(ast, "reward");
    bool found = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::RewardHacking &&
            f.severity == FindingSeverity::ERROR_LEVEL) found = true;
    ASSERT_TRUE(found);
}

TEST(autopsy_reward_hacking_records_cycle_path) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: Start\nSTATE: S2\nSTATE: S3\nSTATE: S4\nSTATE: Goal\n"
        "ACTION: GoToCycle\nACTION: TraverseCycle\nACTION: GoToGoal\nACTION: Stay\n"
        "TRANSITION: Start GoToCycle S2 1.0\nTRANSITION: Start GoToGoal S4 1.0\n"
        "TRANSITION: S2 TraverseCycle S3 1.0\nTRANSITION: S3 TraverseCycle S2 1.0\n"
        "TRANSITION: S2 GoToCycle S2 1.0\nTRANSITION: S2 GoToGoal S2 1.0\nTRANSITION: S2 Stay S2 1.0\n"
        "TRANSITION: S3 GoToCycle S3 1.0\nTRANSITION: S3 GoToGoal S3 1.0\nTRANSITION: S3 Stay S3 1.0\n"
        "TRANSITION: S4 GoToGoal Goal 1.0\nTRANSITION: S4 TraverseCycle S4 1.0\n"
        "TRANSITION: S4 GoToCycle S4 1.0\nTRANSITION: S4 Stay S4 1.0\n"
        "TRANSITION: Goal Stay Goal 1.0\nTRANSITION: Goal TraverseCycle Goal 1.0\n"
        "TRANSITION: Goal GoToCycle Goal 1.0\nTRANSITION: Goal GoToGoal Goal 1.0\n"
        "TRANSITION: Start Stay Start 1.0\nTRANSITION: Start TraverseCycle Start 1.0\n"
        "ACTION_REWARD: S2 TraverseCycle 12.0\nACTION_REWARD: S3 TraverseCycle 12.0\n"
        "ACTION_REWARD: S4 GoToGoal 50.0\nDISCOUNT: 0.95\n");
    auto report = runAutopsy(ast, "reward");
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::RewardHacking) {
            ASSERT_TRUE(!f.text_fields.at("cycle_path").empty());
            ASSERT_TRUE(f.numeric_fields.at("cycle_length") == 2);
        }
}

TEST(autopsy_reward_hacking_does_not_fire_clean) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 10.0\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "reward");
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::RewardHacking);
}

// -------- Class 6: Fragility --------
TEST(autopsy_fragility_reports_some_decision) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: C\n"
        "ACTION: Left\nACTION: Right\n"
        "TRANSITION: A Right B 1.0\nTRANSITION: A Left C 1.0\n"
        "TRANSITION: B Right B 1.0\nTRANSITION: B Left B 1.0\n"
        "TRANSITION: C Right C 1.0\nTRANSITION: C Left C 1.0\n"
        "REWARD: B 10.0\nREWARD: C 9.0\n"
        "DISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "landscape");
    bool found = false;
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::Fragility) {
            ASSERT_TRUE(f.numeric_fields.count("min_flip_delta") > 0);
            ASSERT_TRUE(f.numeric_fields.count("robustness_score") > 0);
            found = true;
        }
    ASSERT_TRUE(found);
}

TEST(autopsy_fragility_high_score_when_rewards_separated) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: GoalHi\nSTATE: GoalLo\n"
        "ACTION: ToHi\nACTION: ToLo\n"
        "TRANSITION: A ToHi GoalHi 1.0\nTRANSITION: A ToLo GoalLo 1.0\n"
        "TRANSITION: GoalHi ToHi GoalHi 1.0\nTRANSITION: GoalHi ToLo GoalHi 1.0\n"
        "TRANSITION: GoalLo ToHi GoalLo 1.0\nTRANSITION: GoalLo ToLo GoalLo 1.0\n"
        "REWARD: GoalHi 1000.0\nREWARD: GoalLo 1.0\n"
        "DISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "landscape");
    for (const auto& f : report.findings)
        if (f.cls == FindingClass::Fragility) {
            // With a 1000:1 reward separation, the flip-delta should be large.
            ASSERT_TRUE(f.numeric_fields.at("min_flip_delta") > 0.1);
        }
}

// -------- Dispatcher modes --------
TEST(autopsy_mode_reward_skips_structural_passes) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: Dead\nACTION: G\n"
        "TRANSITION: A G B 1.0\nTRANSITION: B G B 1.0\n"
        "TRANSITION: Dead G Dead 1.0\n"
        "REWARD: B 10.0\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "reward");
    // 'Dead' is a dead state, but reward-mode should NOT include that finding.
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::DeadState);
}

TEST(autopsy_mode_structural_skips_reward_passes) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: G\n"
        "TRANSITION: A G B 1.0\nTRANSITION: B G B 1.0\n"
        "REWARD: B 1000.0\nREWARD: A 0.001\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "structural");
    for (const auto& f : report.findings)
        ASSERT_TRUE(f.cls != FindingClass::MagnitudeImbalance);
}

TEST(autopsy_full_runs_every_class) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: Dead\nACTION: G\n"
        "TRANSITION: A G B 1.0\nTRANSITION: B G B 1.0\nTRANSITION: Dead G Dead 1.0\n"
        "REWARD: B 1000.0\nREWARD: A 0.001\nDISCOUNT: 0.9\n");
    auto report = runAutopsy(ast, "full");
    bool saw_mag = false, saw_dead = false;
    for (const auto& f : report.findings) {
        if (f.cls == FindingClass::MagnitudeImbalance) saw_mag  = true;
        if (f.cls == FindingClass::DeadState)          saw_dead = true;
    }
    ASSERT_TRUE(saw_mag);
    ASSERT_TRUE(saw_dead);
}

// -------- Idempotency (running twice = same findings) --------
TEST(autopsy_is_deterministic) {
    SilentCout s;
    auto ast1 = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: G\n"
        "TRANSITION: A G B 1.0\nTRANSITION: B G B 1.0\n"
        "REWARD: B 1000.0\nREWARD: A 0.001\nDISCOUNT: 0.9\n");
    auto ast2 = ast1;
    auto r1 = runAutopsy(ast1, "reward");
    auto r2 = runAutopsy(ast2, "reward");
    ASSERT_EQ((int)r1.findings.size(), (int)r2.findings.size());
    for (size_t i = 0; i < r1.findings.size(); ++i) {
        ASSERT_EQ((int)r1.findings[i].cls, (int)r2.findings[i].cls);
        ASSERT_EQ((int)r1.findings[i].severity, (int)r2.findings[i].severity);
    }
}

// -------- Severity API --------
TEST(autopsy_has_issues_when_warn_only) {
    AutopsyReport r;
    AutopsyFinding f;
    f.cls = FindingClass::MagnitudeImbalance;
    f.severity = FindingSeverity::WARN;
    r.findings.push_back(f);
    ASSERT_TRUE(r.has_issues());
    ASSERT_FALSE(r.has_errors());
}

TEST(autopsy_has_errors_when_error_present) {
    AutopsyReport r;
    AutopsyFinding f;
    f.cls = FindingClass::DeadState;
    f.severity = FindingSeverity::ERROR_LEVEL;
    r.findings.push_back(f);
    ASSERT_TRUE(r.has_issues());
    ASSERT_TRUE(r.has_errors());
}

// -------- REPAIR assertion parser --------
TEST(repair_parse_ascii_pi_eq) {
    auto p = parseRepairTarget("pi(Start) == Move");
    ASSERT_EQ((int)p.kind, (int)ParsedAssertion::PI_EQ);
    ASSERT_EQ(p.state, std::string("Start"));
    ASSERT_EQ(p.action, std::string("Move"));
}

TEST(repair_parse_ascii_pi_neq) {
    auto p = parseRepairTarget("pi(Start) != Stay");
    ASSERT_EQ((int)p.kind, (int)ParsedAssertion::PI_NEQ);
    ASSERT_EQ(p.state, std::string("Start"));
    ASSERT_EQ(p.action, std::string("Stay"));
}

TEST(repair_parse_unicode_pi) {
    // UTF-8 for π is CF 80.
    std::string t = std::string("\xCF\x80(Start) == Move");
    auto p = parseRepairTarget(t);
    ASSERT_EQ((int)p.kind, (int)ParsedAssertion::PI_EQ);
}

TEST(repair_parse_garbage_returns_unsupported) {
    auto p = parseRepairTarget("foobar");
    ASSERT_EQ((int)p.kind, (int)ParsedAssertion::UNSUPPORTED);
}

// -------- REPAIR command --------
TEST(repair_already_satisfied_no_change) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 100.0\nDISCOUNT: 0.9\n");
    auto r = repairMDP(ast, "pi(A) == Go");
    ASSERT_TRUE(r.already_satisfied);
    ASSERT_TRUE(r.proposals.empty());
}

TEST(repair_finds_a_single_param_fix) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: Start\nSTATE: Middle\nSTATE: Goal\n"
        "ACTION: Move\nACTION: Stay\n"
        "TRANSITION: Start Move Middle 1.0\nTRANSITION: Start Stay Start 1.0\n"
        "TRANSITION: Middle Move Goal 1.0\nTRANSITION: Middle Stay Middle 1.0\n"
        "TRANSITION: Goal Move Goal 1.0\nTRANSITION: Goal Stay Goal 1.0\n"
        "REWARD: Goal 100.0\n"
        "ACTION_REWARD: Start Move -5.0\nACTION_REWARD: Middle Move -5.0\n"
        "DISCOUNT: 0.15\n");
    auto r = repairMDP(ast, "pi(Start) == Move");
    ASSERT_FALSE(r.already_satisfied);
    ASSERT_TRUE(r.any_succeeded);
    ASSERT_TRUE(!r.proposals.empty());
    // The minimal fix should be a small magnitude.
    ASSERT_TRUE(r.proposals[0].magnitude < r.proposals.back().magnitude + 1e-9);
}

TEST(repair_returns_no_fix_when_target_is_undeclared_state) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nACTION: G\nTRANSITION: A G A 1.0\nREWARD: A 1.0\nDISCOUNT: 0.9\n");
    auto r = repairMDP(ast, "pi(NoSuchState) == G");
    ASSERT_FALSE(r.already_satisfied);
    ASSERT_FALSE(r.any_succeeded);
}

void run_autopsy_tests() {
    std::cout << "\n--- REWARD AUTOPSY TESTS (v3.0 Phase 3A) ---\n";
    RUN_TEST(autopsy_magnitude_error_below_0p1pct);
    RUN_TEST(autopsy_magnitude_warn_between_0p1_and_1pct);
    RUN_TEST(autopsy_magnitude_clean_no_findings);
    RUN_TEST(autopsy_myopia_fires_long_corridor_low_gamma);
    RUN_TEST(autopsy_myopia_does_not_fire_high_gamma);
    RUN_TEST(autopsy_myopia_suggests_gamma_above_current);
    RUN_TEST(autopsy_dead_state_detects_isolated_pair);
    RUN_TEST(autopsy_dead_state_does_not_fire_clean);
    RUN_TEST(autopsy_discount_cliff_fires_when_policy_flips);
    RUN_TEST(autopsy_discount_cliff_records_gamma_and_state);
    RUN_TEST(autopsy_reward_hacking_detects_attractive_cycle);
    RUN_TEST(autopsy_reward_hacking_records_cycle_path);
    RUN_TEST(autopsy_reward_hacking_does_not_fire_clean);
    RUN_TEST(autopsy_fragility_reports_some_decision);
    RUN_TEST(autopsy_fragility_high_score_when_rewards_separated);
    RUN_TEST(autopsy_mode_reward_skips_structural_passes);
    RUN_TEST(autopsy_mode_structural_skips_reward_passes);
    RUN_TEST(autopsy_full_runs_every_class);
    RUN_TEST(autopsy_is_deterministic);
    RUN_TEST(autopsy_has_issues_when_warn_only);
    RUN_TEST(autopsy_has_errors_when_error_present);
    RUN_TEST(repair_parse_ascii_pi_eq);
    RUN_TEST(repair_parse_ascii_pi_neq);
    RUN_TEST(repair_parse_unicode_pi);
    RUN_TEST(repair_parse_garbage_returns_unsupported);
    RUN_TEST(repair_already_satisfied_no_change);
    RUN_TEST(repair_finds_a_single_param_fix);
    RUN_TEST(repair_returns_no_fix_when_target_is_undeclared_state);
}


// ============================================================================
// v3.0 PHASE 3B — BACKWARDS SOLVER / MDP_AUTOPSY TESTS
// ============================================================================
// Tests cover:
//   - .log file parser (header, FAILURE marker, step rows, validation)
//   - Empirical transition counting (absorbing-state skipping)
//   - KL-divergence ranking (PRIMARY SUSPECT identification)
//   - Verdict classifier (ACCURATE / OPTIMISTIC / DANGEROUS thresholds)
//   - Backwards solver (already-fails, never-fails, intermediate-alpha)
// ============================================================================

// Helper: write a log to a temp path so parseLogFile can read it.
static std::string writeTempLog(const std::string& body) {
    static int counter = 0;
    std::string path = "/tmp/mdp_log_test_" + std::to_string(counter++) + ".log";
    std::ofstream f(path); f << body; f.close();
    return path;
}

TEST(log_parser_accepts_simple_log) {
    SilentCout s;
    std::string path = writeTempLog(
        "LOG:\n"
        "  step 1 state A action Go reward 0\n"
        "  step 2 state B action Go reward 1\n"
        "  FAILURE\n");
    ExecutionLog log;
    std::string err;
    ASSERT_TRUE(parseLogFile(path, log, err));
    ASSERT_EQ((int)log.steps.size(), 2);
    ASSERT_TRUE(log.failed);
    ASSERT_EQ(log.steps[0].state, std::string("A"));
    ASSERT_EQ(log.steps[1].state, std::string("B"));
}

TEST(log_parser_no_failure_marker) {
    SilentCout s;
    std::string path = writeTempLog(
        "LOG:\n"
        "  step 1 state A action Go reward 0\n");
    ExecutionLog log; std::string err;
    ASSERT_TRUE(parseLogFile(path, log, err));
    ASSERT_FALSE(log.failed);
}

TEST(log_parser_skips_comments_and_blanks) {
    SilentCout s;
    std::string path = writeTempLog(
        "# this is a comment\n"
        "\n"
        "LOG:\n"
        "# inline comment\n"
        "\n"
        "  step 1 state A action Go reward 0\n"
        "  step 2 state B action Go reward 1\n"
        "  FAILURE\n");
    ExecutionLog log; std::string err;
    ASSERT_TRUE(parseLogFile(path, log, err));
    ASSERT_EQ((int)log.steps.size(), 2);
}

TEST(log_parser_rejects_malformed_line) {
    SilentCout s;
    std::string path = writeTempLog(
        "LOG:\n"
        "  this line has no state or action\n");
    ExecutionLog log; std::string err;
    ASSERT_FALSE(parseLogFile(path, log, err));
    ASSERT_FALSE(err.empty());
}

TEST(log_parser_missing_file) {
    SilentCout s;
    ExecutionLog log; std::string err;
    ASSERT_FALSE(parseLogFile("/tmp/__no_such_log_file__.log", log, err));
}

TEST(log_validate_unknown_state_rejected) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go B 1.0\nTRANSITION: B Go B 1.0\n"
        "REWARD: B 1.0\nDISCOUNT: 0.9\n");
    ExecutionLog log;
    log.steps.push_back({1, "A", "Go", 0.0});
    log.steps.push_back({2, "Z", "Go", 0.0});   // Z is unknown
    std::string err;
    ASSERT_FALSE(validateLogAgainstAST(log, ast, err));
}

TEST(log_validate_unknown_action_rejected) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nACTION: Go\nTRANSITION: A Go A 1.0\nREWARD: A 1.0\nDISCOUNT: 0.9\n");
    ExecutionLog log;
    log.steps.push_back({1, "A", "Fly", 0.0});
    std::string err;
    ASSERT_FALSE(validateLogAgainstAST(log, ast, err));
}

TEST(empirical_counts_transitions) {
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nSTATE: C\nACTION: Go\n"
        "TRANSITION: A Go A 0.5\nTRANSITION: A Go B 0.5\n"
        "TRANSITION: B Go A 0.5\nTRANSITION: B Go C 0.5\n"
        "TRANSITION: C Go C 1.0\n"
        "REWARD: C 1.0\nDISCOUNT: 0.9\n");
    // Log: A->B->A->B - empirical should count 2 (A,Go) and 1 (B,Go).
    // C is absorbing; we don't transition through it here.
    ExecutionLog log;
    log.steps = {{1,"A","Go",0},{2,"B","Go",0},{3,"A","Go",0},{4,"B","Go",0}};
    auto emp = buildEmpirical(log, &ast);
    ASSERT_EQ(emp.total.at("A").at("Go"), 2);
    ASSERT_EQ(emp.total.at("B").at("Go"), 1);
    ASSERT_EQ(emp.count.at("A").at("Go").at("B"), 2);
}

TEST(empirical_skips_absorbing_state_transitions) {
    // Absorbing state Z must not contribute empirical transitions even if
    // the log shows transitions out of it (mission resets).
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: Z\nACTION: Go\n"
        "TRANSITION: A Go A 1.0\nTRANSITION: Z Go Z 1.0\n"
        "REWARD: A 1.0\nDISCOUNT: 0.9\n");
    ExecutionLog log;
    log.steps = {{1,"A","Go",0},{2,"Z","Go",0},{3,"A","Go",0}};
    auto emp = buildEmpirical(log, &ast);
    // The (Z, Go) -> A transition (a reset) must be skipped.
    ASSERT_EQ(emp.total.count("Z"), 0u);
}

TEST(kl_ranks_most_divergent_pair_first) {
    SilentCout s;
    // Model says (A, Go) -> A is 99%; log shows (A, Go) -> B is 100%.
    // (B, Go) goes to A; matches model perfectly. Ranking puts (A, Go) on top.
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\n"
        "TRANSITION: A Go A 0.99\nTRANSITION: A Go B 0.01\n"
        "TRANSITION: B Go A 1.0\n"
        "REWARD: B 1.0\nDISCOUNT: 0.9\n");
    ExecutionLog log;
    int t = 0;
    for (int i = 0; i < 30; ++i) {
        log.steps.push_back({++t, "A", "Go", 0});
        log.steps.push_back({++t, "B", "Go", 0});
    }
    auto emp = buildEmpirical(log, &ast);
    auto rows = rankByKL(ast, emp);
    ASSERT_TRUE(!rows.empty());
    ASSERT_EQ(rows[0].state, std::string("A"));
    ASSERT_EQ(rows[0].action, std::string("Go"));
}

TEST(verdict_accurate_when_factors_below_2) {
    std::vector<DivergenceRow> rows;
    DivergenceRow r{};
    r.state = "A"; r.action = "Go";
    r.observation_count = 100;
    r.worst_underestimation_factor = 1.5;
    rows.push_back(r);
    ASSERT_EQ((int)classifyVerdict(rows), (int)Verdict::ACCURATE);
}

TEST(verdict_optimistic_when_factor_2_to_10) {
    std::vector<DivergenceRow> rows;
    DivergenceRow r{};
    r.state = "A"; r.action = "Go";
    r.observation_count = 100;
    r.worst_underestimation_factor = 5.0;
    rows.push_back(r);
    ASSERT_EQ((int)classifyVerdict(rows), (int)Verdict::OPTIMISTIC);
}

TEST(verdict_dangerous_when_factor_above_10) {
    std::vector<DivergenceRow> rows;
    DivergenceRow r{};
    r.state = "A"; r.action = "Go";
    r.observation_count = 100;
    r.worst_underestimation_factor = 15.0;
    rows.push_back(r);
    ASSERT_EQ((int)classifyVerdict(rows), (int)Verdict::DANGEROUS);
}

TEST(verdict_ignores_low_confidence_rows) {
    // A row with 12.5x factor but only 5 observations should be ignored
    // by the verdict classifier (falls below the 30-obs threshold; the
    // fallback 10-obs threshold also excludes it).
    std::vector<DivergenceRow> rows;
    DivergenceRow r{};
    r.state = "A"; r.action = "Go";
    r.observation_count = 5;
    r.worst_underestimation_factor = 12.5;
    rows.push_back(r);
    ASSERT_EQ((int)classifyVerdict(rows), (int)Verdict::ACCURATE);
}

TEST(backwards_solver_no_failure_when_model_robust) {
    // Model where the VERIFY block is much looser than the value attained,
    // so no empirical-direction perturbation breaks it. Use two states so
    // A isn't classified as absorbing.
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: B\nACTION: Go\nACTION: Stay\n"
        "TRANSITION: A Go   A 0.5\nTRANSITION: A Go   B 0.5\n"
        "TRANSITION: A Stay A 1.0\n"
        "TRANSITION: B Go   B 0.5\nTRANSITION: B Go   A 0.5\n"
        "TRANSITION: B Stay B 1.0\n"
        "REWARD: A 1.0\nREWARD: B 1.0\nDISCOUNT: 0.9\n"
        "VERIFY: V(A) > -1000.0\n");
    ExecutionLog log;
    for (int i = 0; i < 30; ++i) {
        log.steps.push_back({2*i+1,"A","Go",0});
        log.steps.push_back({2*i+2,"B","Go",0});
    }
    auto emp = buildEmpirical(log, &ast);
    auto out = runBackwardsSolver(ast, emp);
    ASSERT_FALSE(out.found);     // model is robust
}

TEST(backwards_solver_finds_intermediate_alpha) {
    // We need the optimal policy to USE the action whose transitions
    // we're perturbing. So Stop is made worse than Go: Stop has zero
    // reward at A and B, while A->A Go gets the +1 reward.
    SilentCout s;
    auto ast = parseMDPString(
        "STATE: A\nSTATE: Bad\nACTION: Go\nACTION: Stop\n"
        "TRANSITION: A   Go   A   0.9\nTRANSITION: A   Go   Bad 0.1\n"
        "TRANSITION: A   Stop A   1.0\n"
        "TRANSITION: Bad Go   Bad 0.9\nTRANSITION: Bad Go   A   0.1\n"
        "TRANSITION: Bad Stop Bad 1.0\n"
        "REWARD: A 1.0\nREWARD: Bad -100.0\n"
        "ACTION_REWARD: A Stop -2.0\n"        // Stop is worse than Go at A
        "ACTION_REWARD: Bad Stop -1.0\n"      // Stop is worse than Go at Bad too
        "DISCOUNT: 0.9\n"
        "VERIFY: V(A) > -5.0\n");

    // Build an empirical log: 40 (A, Go) -> Bad of 40 observations.
    ExecutionLog log;
    for (int i = 0; i < 40; ++i) {
        log.steps.push_back({2*i+1, "A",   "Go", 0});
        log.steps.push_back({2*i+2, "Bad", "Go", 0});
    }
    auto emp = buildEmpirical(log, &ast);
    auto out = runBackwardsSolver(ast, emp);
    ASSERT_TRUE(out.found);
    ASSERT_TRUE(out.alpha > 0.0 && out.alpha < 1.0);
    ASSERT_TRUE(out.l1_norm > 0.0);
}

void run_phase3b_tests() {
    std::cout << "\n--- PHASE 3B: BACKWARDS SOLVER / mdp_autopsy TESTS ---\n";
    RUN_TEST(log_parser_accepts_simple_log);
    RUN_TEST(log_parser_no_failure_marker);
    RUN_TEST(log_parser_skips_comments_and_blanks);
    RUN_TEST(log_parser_rejects_malformed_line);
    RUN_TEST(log_parser_missing_file);
    RUN_TEST(log_validate_unknown_state_rejected);
    RUN_TEST(log_validate_unknown_action_rejected);
    RUN_TEST(empirical_counts_transitions);
    RUN_TEST(empirical_skips_absorbing_state_transitions);
    RUN_TEST(kl_ranks_most_divergent_pair_first);
    RUN_TEST(verdict_accurate_when_factors_below_2);
    RUN_TEST(verdict_optimistic_when_factor_2_to_10);
    RUN_TEST(verdict_dangerous_when_factor_above_10);
    RUN_TEST(verdict_ignores_low_confidence_rows);
    RUN_TEST(backwards_solver_no_failure_when_model_robust);
    RUN_TEST(backwards_solver_finds_intermediate_alpha);
}


// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  MDP-DSL v3.0 — Test Suite\n";
    std::cout << "========================================\n";

    run_parser_tests();
    run_validator_tests();
    run_vi_tests();
    run_pi_tests();
    run_ql_tests();
    run_integration_tests();
    run_type_tests();
    run_macro_tests();
    run_verify_tests();

    // v3.0 — POMDP / PBVI test groups
    run_pomdp_parser_tests();
    run_pomdp_validator_tests();
    run_belief_update_tests();
    run_pbvi_tests();
    run_pomdp_integration_tests();

    // v3.0 Phase 2A — HMM-MDP bridge tests
    run_hmm_tests();

    // v3.0 Phase 3A — Reward Autopsy Engine tests
    run_autopsy_tests();

    // v3.0 Phase 3B — Backwards Solver / mdp_autopsy tests
    run_phase3b_tests();

    std::cout << "\n========================================\n";
    std::cout << "  TOTAL: " << tf::passes << "/" << (tf::passes + tf::failures) << " assertions passed\n";
    if (tf::failures == 0)
        std::cout << "  ALL TESTS PASSED\n";
    else
        std::cout << "  " << tf::failures << " FAILURES\n";
    std::cout << "========================================\n";
    return tf::failures > 0 ? 1 : 0;
}
