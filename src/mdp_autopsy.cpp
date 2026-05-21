// ============================================================================
// mdp_autopsy.cpp — MDP-DSL v3.0 Phase 3B
//
// Failure-log backwards solver.
//
// Given a compiled .mdp model and an execution log that ended in FAILURE,
// computes:
//   (1) The empirical transition frequencies T_empirical(s,a,s') from the log.
//   (2) The (s,a) pair where T_empirical diverges most from T_model under KL.
//       This is the PRIMARY SUSPECT.
//   (3) The minimum perturbation ||ΔT||_1 (along the empirical direction)
//       under which at least one VERIFY assertion in the model would fail.
//   (4) A verdict: MODEL_ACCURATE / MODEL_OPTIMISTIC / MODEL_DANGEROUS,
//       graded by how badly the worst (s,a) pair underestimates the truth.
//
// This is conceptually distinct from Phase 3A's --diagnose command:
//   3A diagnoses *model design errors* (the .mdp is internally pathological).
//   3B diagnoses *model-reality divergence* (the .mdp was internally fine,
//      but the world disagreed with its transition probabilities).
//
// CLI:
//   ./mdp_autopsy model.mdp run.log
//   ./mdp_autopsy model.mdp run.log --report-only   # skip backwards solver
//   ./mdp_autopsy model.mdp run.log --json          # machine-readable
//
// Exit codes:
//   0 = MODEL_ACCURATE
//   1 = MODEL_OPTIMISTIC
//   2 = MODEL_DANGEROUS
//   3 = parse error / bad log
//
// Single C++17 translation unit. Zero external dependencies.
// Pulls in the entire compiler via #include "mdp_compiler.cpp".
//
// Author: Charvit Rajani, IIT Guwahati (Roll No. 240102028).
// ============================================================================

// Pull in the compiler library WITHOUT its main(). The MDP_TESTING_MODE guard
// in mdp_compiler.cpp excludes main() and the viz includes.
#define MDP_TESTING_MODE
#include "mdp_compiler.cpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// SECTION A: LOG FILE FORMAT + PARSER
// ============================================================================
//
// Expected format (case-insensitive; blank lines and #-comments are ignored):
//
//   LOG:
//     step 1  state <name>  action <name>  reward <float>
//     step 2  state <name>  action <name>  reward <float>
//     ...
//     step N  state <name>  action <name>  reward <float>
//     FAILURE
//
// The trailing FAILURE marker is optional. When absent, the report classifies
// the run as a NON-FAILURE divergence report (model vs reality, no minimum
// edit needed).
//
// Validation:
//   * Every state / action name must exist in the AST.
//   * Steps form an implicit (s_t, a_t, s_{t+1}) triple: the transition is
//     from step t's state to step (t+1)'s state, via step t's action.
//   * Unknown names emit ERR_LOG (exit 3).
// ============================================================================

struct LogStep {
    int         step;
    std::string state;
    std::string action;
    double      reward;
};

struct ExecutionLog {
    std::vector<LogStep> steps;
    bool                 failed = false;   // true iff trailing FAILURE marker present
    std::string          source_file;
};

// Trim ASCII whitespace from both ends.
static std::string lt_trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
    return s.substr(a, b - a);
}

static std::string lt_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Parse a .log file. Returns false on malformed input; errors written to `err`.
static bool parseLogFile(const std::string& path, ExecutionLog& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "Could not open log file: " + path; return false; }
    out.source_file = path;
    std::string line;
    bool in_log = false;
    int  line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        std::string trimmed = lt_trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        std::string lower = lt_lower(trimmed);

        // Header line: "LOG:"
        if (!in_log) {
            if (lower == "log:" || lower == "log") { in_log = true; continue; }
            // Allow logs without a header.
            in_log = true;
            // Fall through to step-parsing below.
        }

        // FAILURE marker.
        if (lower == "failure" || lower == "failure:") {
            out.failed = true;
            continue;
        }

        // Step line: "step <N> state <name> action <name> reward <float>"
        // We parse with a forgiving scanner: skip non-keyword tokens, then
        // look for the four expected keyword/value pairs. The parse SUCCEEDS
        // only when both state and action come out non-empty.
        std::istringstream ls(trimmed);
        std::string tok;
        LogStep step{};
        step.step   = -1;
        step.reward = 0.0;
        bool have_state = false, have_action = false;
        while (ls >> tok) {
            std::string key = lt_lower(tok);
            if (key == "step") { ls >> step.step; }
            else if (key == "state")  {
                std::string v;
                if (ls >> v && !v.empty()) { step.state = v; have_state = true; }
            }
            else if (key == "action") {
                std::string v;
                if (ls >> v && !v.empty()) { step.action = v; have_action = true; }
            }
            else if (key == "reward") { ls >> step.reward; }
        }
        if (!have_state || !have_action) {
            std::ostringstream oss;
            oss << "Malformed log line " << line_no << ": '" << trimmed << "'";
            err = oss.str();
            return false;
        }
        if (step.step < 0) step.step = static_cast<int>(out.steps.size()) + 1;
        out.steps.push_back(step);
    }
    if (out.steps.empty()) { err = "Log contains no step entries."; return false; }
    return true;
}

// Validate every state/action mentioned in the log exists in the model.
static bool validateLogAgainstAST(const ExecutionLog& log, const MDP_AST& ast,
                                  std::string& err) {
    for (const auto& step : log.steps) {
        if (!ast.states.count(step.state)) {
            err = "Log references unknown state '" + step.state + "' at step "
                  + std::to_string(step.step) + ".";
            return false;
        }
        if (!ast.actions.count(step.action)) {
            err = "Log references unknown action '" + step.action + "' at step "
                  + std::to_string(step.step) + ".";
            return false;
        }
    }
    return true;
}

// ============================================================================
// SECTION B: EMPIRICAL TRANSITION COUNTING + KL DIVERGENCE
// ============================================================================
//
// For consecutive steps t and t+1, the transition (s_t, a_t) -> s_{t+1} is
// recorded. We build:
//   count[s][a][s_prime] : raw observation count
//   total[s][a]          : total (s,a) observations
// Then T_emp(s,a,s') = count / total (or NaN if total < 1).
//
// We rank (s,a) pairs by KL(T_emp || T_model), with additive smoothing
// epsilon = 1e-10 where T_model = 0 to keep KL finite.
// ============================================================================

struct EmpiricalTransitions {
    // count[s][a][s'] -> raw count
    std::unordered_map<std::string,
        std::unordered_map<std::string,
            std::unordered_map<std::string, int>>> count;
    // total[s][a] -> total observations for that (s,a) pair
    std::unordered_map<std::string,
        std::unordered_map<std::string, int>> total;
};

static EmpiricalTransitions buildEmpirical(const ExecutionLog& log,
                                           const MDP_AST* ast_for_absorbing = nullptr) {
    // A state s is "absorbing" in the model if for every action a, the row
    // (s, a) has p=1.0 only to (s, a, s). Empirical events leaving an absorbing
    // state (such as "robot was reset to Safe after collision") are physical
    // interventions, not transitions of the modeled MDP. We skip them so the
    // KL divergence computation isn't contaminated by reset behavior.
    std::unordered_set<std::string> absorbing;
    if (ast_for_absorbing) {
        for (const auto& s : ast_for_absorbing->states) {
            bool all_self = true;
            bool any_seen = false;
            for (const auto& a : ast_for_absorbing->actions) {
                bool has_row = false, only_self_p1 = false;
                for (const auto& t : ast_for_absorbing->transitions) {
                    if (t.source_state == s && t.action == a) {
                        has_row = true;
                        if (t.dest_state == s && std::abs(t.probability - 1.0) < 1e-9) {
                            only_self_p1 = true;
                        } else if (t.probability > 0) {
                            only_self_p1 = false; break;
                        }
                    }
                }
                if (has_row) any_seen = true;
                if (has_row && !only_self_p1) { all_self = false; break; }
            }
            if (any_seen && all_self) absorbing.insert(s);
        }
    }

    EmpiricalTransitions emp;
    for (size_t t = 0; t + 1 < log.steps.size(); ++t) {
        const auto& cur = log.steps[t];
        const auto& nxt = log.steps[t + 1];
        if (absorbing.count(cur.state)) continue;  // skip reset events
        emp.count[cur.state][cur.action][nxt.state] += 1;
        emp.total[cur.state][cur.action] += 1;
    }
    return emp;
}

// Look up T_model(s, a, s'). Returns 0 if no row for (s, a, s') exists.
static double modelProb(const MDP_AST& ast, const std::string& s,
                        const std::string& a, const std::string& sp) {
    for (const auto& t : ast.transitions) {
        if (t.source_state == s && t.action == a && t.dest_state == sp)
            return t.probability;
    }
    return 0.0;
}

struct DivergenceRow {
    std::string state;
    std::string action;
    int         observation_count;
    double      kl_divergence;
    // For reporting: the worst single (s,a,s') triple in this pair.
    std::string worst_dest;
    double      worst_model_p;
    double      worst_empirical_p;
    double      worst_underestimation_factor;  // empirical / max(model, eps)
};

// Compute KL(emp || model) per (s,a) pair, with smoothing.
static std::vector<DivergenceRow> rankByKL(const MDP_AST& ast,
                                           const EmpiricalTransitions& emp,
                                           int low_confidence_threshold = 30) {
    constexpr double EPS = 1e-10;
    std::vector<DivergenceRow> rows;

    // Build the set of all destination states the model knows about, per (s,a).
    // We need this for the smoothing across the model's full support.
    for (const auto& [s, am] : emp.count) {
        for (const auto& [a, dest_counts] : am) {
            int total = emp.total.at(s).at(a);
            if (total < 1) continue;

            // Union of model destinations and empirical destinations.
            std::set<std::string> dests;
            for (const auto& t : ast.transitions)
                if (t.source_state == s && t.action == a) dests.insert(t.dest_state);
            for (const auto& [sp, _] : dest_counts) dests.insert(sp);

            DivergenceRow row{};
            row.state = s; row.action = a;
            row.observation_count = total;
            row.kl_divergence = 0.0;
            row.worst_underestimation_factor = 0.0;
            row.worst_model_p = 0.0;
            row.worst_empirical_p = 0.0;

            for (const auto& sp : dests) {
                double emp_p   = (dest_counts.count(sp) ? dest_counts.at(sp) : 0)
                                  / static_cast<double>(total);
                double model_p = modelProb(ast, s, a, sp);
                double emp_s   = std::max(emp_p,   EPS);
                double model_s = std::max(model_p, EPS);
                row.kl_divergence += emp_s * std::log(emp_s / model_s);

                // Underestimation factor: by how much did empirical exceed model?
                // If the model assigned essentially zero probability, the
                // factor is unbounded; cap it at the smaller-but-still-large
                // value 1000 for ranking purposes, so a single never-modeled
                // transition doesn't dominate the report with billion-x numbers.
                double factor;
                if (model_p < 1e-6) {
                    factor = (emp_p > 0) ? 1000.0 : 0.0;
                } else {
                    factor = emp_p / model_p;
                }
                if (emp_p > model_p && factor > row.worst_underestimation_factor) {
                    row.worst_underestimation_factor = factor;
                    row.worst_dest        = sp;
                    row.worst_model_p     = model_p;
                    row.worst_empirical_p = emp_p;
                }
            }
            rows.push_back(row);
        }
    }
    // Sort by KL descending. Low-confidence pairs (fewer than 30 obs) get
    // pushed to the bottom because their KL is noisy.
    std::sort(rows.begin(), rows.end(),
              [low_confidence_threshold](const DivergenceRow& a, const DivergenceRow& b) {
                  bool a_low = a.observation_count < low_confidence_threshold;
                  bool b_low = b.observation_count < low_confidence_threshold;
                  if (a_low != b_low) return !a_low;  // confident pairs first
                  return a.kl_divergence > b.kl_divergence;
              });
    return rows;
}

// ============================================================================
// SECTION C: BACKWARDS SOLVER
// ============================================================================
//
// Given the model AST, its VERIFY assertions, and the empirical transitions,
// search for the minimum-magnitude perturbation ΔT (along the empirical
// direction) such that at least one VERIFY assertion fails under T + ΔT.
//
// The empirical direction:
//   For each (s, a) with at least one observation, define
//   d(s,a,s') = T_emp(s,a,s') - T_model(s,a,s').
//   Then for scaling factor alpha ∈ [0, 1]:
//     T_alpha(s,a,s') = T_model + alpha * d(s,a,s').
//   At alpha = 0: we are at the model (assumptions all hold).
//   At alpha = 1: we are at the empirical distribution.
//
// We binary-search alpha for the smallest value where at least one VERIFY
// assertion fails. The L1 magnitude of the perturbation is
//   ||ΔT||_1 = alpha * sum_{s,a,s'} |d(s,a,s')|.
//
// Honest scope note: the master spec calls for a free subgradient descent
// over the simplex. We implement linear search along the empirical direction
// because (a) the empirical data is exactly the right direction to probe in,
// and (b) free optimization over the constrained simplex is hard without a
// QP solver, which would violate the zero-dependency constraint. The
// linear-search formulation honors the spec's "minimum-magnitude perturbation
// that violates a VERIFY" semantics while remaining tractable in pure C++17.
// ============================================================================

struct BackwardsSolverResult {
    bool                  found;
    double                alpha;            // scaling along empirical direction
    double                l1_norm;          // total ||ΔT||_1
    std::string           triggering_assertion;
    // Per-(s,a,s') the perturbation we ended up applying:
    std::vector<std::tuple<std::string, std::string, std::string, double>> deltas;
};

// Apply an empirical-direction perturbation of scale alpha to the AST's
// transitions. Returns a *new* AST with modified transition probabilities.
static MDP_AST applyAlphaPerturbation(const MDP_AST& ast,
                                      const EmpiricalTransitions& emp,
                                      double alpha)
{
    MDP_AST out = ast;
    // For each (s, a) with empirical data, scale model -> empirical by alpha.
    // We replace every existing transition row for that (s, a) and add new
    // rows for any destinations that the empirical distribution observed but
    // the model didn't have a row for.
    std::unordered_set<std::string> handled_pairs;

    for (const auto& [s, am] : emp.total) {
        for (const auto& [a, total] : am) {
            if (total < 1) continue;
            std::string pair_key = s + "|" + a;
            handled_pairs.insert(pair_key);

            // Build the empirical distribution over the union of dests.
            std::set<std::string> dests;
            for (const auto& t : ast.transitions)
                if (t.source_state == s && t.action == a) dests.insert(t.dest_state);
            const auto& emp_dests = emp.count.at(s).at(a);
            for (const auto& [sp, _] : emp_dests) dests.insert(sp);

            // Compute new probabilities for each destination.
            std::unordered_map<std::string, double> new_probs;
            double sum_for_normalization = 0.0;
            for (const auto& sp : dests) {
                double model_p = modelProb(ast, s, a, sp);
                double emp_p   = (emp_dests.count(sp) ? emp_dests.at(sp) : 0)
                                  / static_cast<double>(total);
                double new_p   = (1.0 - alpha) * model_p + alpha * emp_p;
                if (new_p < 0) new_p = 0;
                new_probs[sp] = new_p;
                sum_for_normalization += new_p;
            }
            // Renormalize defensively (handles floating-point drift).
            if (sum_for_normalization > 0) {
                for (auto& [sp, p] : new_probs) p /= sum_for_normalization;
            }

            // Remove all existing rows for (s, a) and add the new ones.
            out.transitions.erase(
                std::remove_if(out.transitions.begin(), out.transitions.end(),
                    [&](const Transition& t){ return t.source_state == s && t.action == a; }),
                out.transitions.end());
            for (const auto& [sp, p] : new_probs) {
                if (p > 0) {
                    Transition tnew;
                    tnew.source_state = s;
                    tnew.action       = a;
                    tnew.dest_state   = sp;
                    tnew.probability  = p;
                    out.transitions.push_back(tnew);
                }
            }
        }
    }
    return out;
}

// Try a single alpha and see if any VERIFY assertion fails.
struct VerifyOutcome {
    bool        any_failed;
    std::string first_failed_assertion;
};
static VerifyOutcome verifyUnderPerturbation(const MDP_AST& ast_orig,
                                             const EmpiricalTransitions& emp,
                                             double alpha)
{
    MDP_AST ast = applyAlphaPerturbation(ast_orig, emp, alpha);

    // Silence the verifier's stdout chatter by temporarily redirecting cout.
    std::stringstream sink;
    std::streambuf* orig_cout = std::cout.rdbuf(sink.rdbuf());

    SolverResult solver_result;
    // Use VI for stability across perturbations.
    std::string saved_mode = ast.solver_mode;
    ast.solver_mode = "value_iteration";
    solver_result = solveValueIteration(ast);
    ast.solver_mode = saved_mode;

    VerifyResult vr = runVerification(solver_result, ast);

    std::cout.rdbuf(orig_cout);

    VerifyOutcome out{};
    out.any_failed = vr.fail_count > 0;
    if (out.any_failed) {
        // VerifyResult.messages contains one entry per assertion, in
        // declaration order. The text begins with "[VERIFY PASS]" or
        // "[VERIFY FAIL]". Find the first failing one and pull the raw
        // assertion text from ast.verify_assertions for clean reporting.
        for (size_t i = 0; i < vr.messages.size(); ++i) {
            if (vr.messages[i].find("[VERIFY FAIL]") != std::string::npos) {
                if (i < ast.verify_assertions.size())
                    out.first_failed_assertion = ast.verify_assertions[i];
                else
                    out.first_failed_assertion = vr.messages[i];
                break;
            }
        }
        // Fallback: if we couldn't isolate the index, just use a generic label.
        if (out.first_failed_assertion.empty() && !ast.verify_assertions.empty())
            out.first_failed_assertion = ast.verify_assertions.front();
    }
    return out;
}

// Compute total L1 norm of empirical - model.
static double computeEmpiricalL1(const MDP_AST& ast,
                                 const EmpiricalTransitions& emp)
{
    double total = 0.0;
    for (const auto& [s, am] : emp.total) {
        for (const auto& [a, n] : am) {
            if (n < 1) continue;
            std::set<std::string> dests;
            for (const auto& t : ast.transitions)
                if (t.source_state == s && t.action == a) dests.insert(t.dest_state);
            const auto& emp_dests = emp.count.at(s).at(a);
            for (const auto& [sp, _] : emp_dests) dests.insert(sp);
            for (const auto& sp : dests) {
                double model_p = modelProb(ast, s, a, sp);
                double emp_p   = (emp_dests.count(sp) ? emp_dests.at(sp) : 0)
                                  / static_cast<double>(n);
                total += std::abs(emp_p - model_p);
            }
        }
    }
    return total;
}

// Binary search the smallest alpha in [0, 1] where some VERIFY assertion fails.
// Returns alpha=1 if no failure even at full empirical perturbation; alpha=0
// if assertion already fails in the unperturbed model.
static BackwardsSolverResult runBackwardsSolver(const MDP_AST& ast,
                                                const EmpiricalTransitions& emp)
{
    BackwardsSolverResult out{};
    out.found  = false;
    out.alpha  = 1.0;
    out.l1_norm = 0.0;

    // First check: does the model already fail VERIFY?
    auto baseline = verifyUnderPerturbation(ast, emp, 0.0);
    if (baseline.any_failed) {
        out.found = true;
        out.alpha = 0.0;
        out.l1_norm = 0.0;
        out.triggering_assertion = baseline.first_failed_assertion;
        return out;
    }
    // Second check: does the model fail at full empirical perturbation?
    auto full = verifyUnderPerturbation(ast, emp, 1.0);
    if (!full.any_failed) {
        // Even believing the empirical distribution fully, no VERIFY breaks.
        // The model is sound w.r.t. the observed reality.
        out.found = false;
        out.alpha = 1.0;
        return out;
    }

    // Binary-search alpha in [0, 1].
    double lo = 0.0, hi = 1.0;
    out.triggering_assertion = full.first_failed_assertion;
    for (int iter = 0; iter < 30; ++iter) {
        double mid = 0.5 * (lo + hi);
        auto r = verifyUnderPerturbation(ast, emp, mid);
        if (r.any_failed) { hi = mid; out.triggering_assertion = r.first_failed_assertion; }
        else              { lo = mid; }
    }
    out.alpha = hi;
    double full_l1 = computeEmpiricalL1(ast, emp);
    out.l1_norm = out.alpha * full_l1;
    out.found = true;

    // Materialize the deltas for reporting.
    for (const auto& [s, am] : emp.total) {
        for (const auto& [a, n] : am) {
            if (n < 1) continue;
            std::set<std::string> dests;
            for (const auto& t : ast.transitions)
                if (t.source_state == s && t.action == a) dests.insert(t.dest_state);
            const auto& emp_dests = emp.count.at(s).at(a);
            for (const auto& [sp, _] : emp_dests) dests.insert(sp);
            for (const auto& sp : dests) {
                double model_p = modelProb(ast, s, a, sp);
                double emp_p   = (emp_dests.count(sp) ? emp_dests.at(sp) : 0)
                                  / static_cast<double>(n);
                double delta = out.alpha * (emp_p - model_p);
                if (std::abs(delta) > 1e-6)
                    out.deltas.emplace_back(s, a, sp, delta);
            }
        }
    }
    return out;
}

// ============================================================================
// SECTION D: VERDICT CLASSIFIER
// ============================================================================
//
// The verdict grades the worst observed underestimation factor:
//   factor < 2     : MODEL_ACCURATE   (within standard noise)
//   factor < 10    : MODEL_OPTIMISTIC (worth investigating)
//   factor >= 10   : MODEL_DANGEROUS  (the model was substantially wrong)
//
// If --report-only was passed (no backwards solver), the verdict is based
// solely on the worst KL row.
// ============================================================================

enum class Verdict { ACCURATE = 0, OPTIMISTIC = 1, DANGEROUS = 2 };

static const char* verdictName(Verdict v) {
    switch (v) {
        case Verdict::ACCURATE:   return "MODEL_ACCURATE";
        case Verdict::OPTIMISTIC: return "MODEL_OPTIMISTIC";
        case Verdict::DANGEROUS:  return "MODEL_DANGEROUS";
    }
    return "UNKNOWN";
}

static Verdict classifyVerdict(const std::vector<DivergenceRow>& rows,
                               int low_confidence_threshold = 30)
{
    double worst_factor = 1.0;
    for (const auto& r : rows) {
        if (r.observation_count < low_confidence_threshold) continue;
        if (r.worst_underestimation_factor > worst_factor)
            worst_factor = r.worst_underestimation_factor;
    }
    // If we had no confident pairs at all, fall back to the unfiltered max
    // BUT require a minimum of 10 observations to avoid noise.
    if (worst_factor == 1.0) {
        for (const auto& r : rows) {
            if (r.observation_count < 10) continue;
            if (r.worst_underestimation_factor > worst_factor)
                worst_factor = r.worst_underestimation_factor;
        }
    }
    if (worst_factor < 2.0)  return Verdict::ACCURATE;
    if (worst_factor < 10.0) return Verdict::OPTIMISTIC;
    return Verdict::DANGEROUS;
}

// ============================================================================
// SECTION E: REPORT PRINTERS
// ============================================================================

static void printHumanReport(const std::string& model_path,
                             const std::string& log_path,
                             const ExecutionLog& log,
                             const std::vector<DivergenceRow>& rows,
                             const BackwardsSolverResult* solver,
                             Verdict verdict)
{
    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << "  MDP AUTOPSY REPORT\n";
    std::cout << "======================================\n";
    std::cout << "  Source model: " << model_path << "\n";
    std::cout << "  Execution log: " << log_path << "\n";
    std::cout << "  Log length: " << log.steps.size() << " steps";
    if (log.failed) std::cout << " (terminated in FAILURE)";
    else            std::cout << " (no FAILURE marker)";
    std::cout << "\n";

    // -- Failure summary --
    if (log.failed && !log.steps.empty()) {
        const auto& last = log.steps.back();
        double cum_reward = 0.0;
        for (const auto& s : log.steps) cum_reward += s.reward;
        std::cout << "\n";
        std::cout << "  FAILURE SUMMARY\n";
        std::cout << "  ---------------\n";
        std::cout << "    Failure at step " << last.step << " in state '" << last.state << "'\n";
        std::cout << "    Last action: " << last.action << "\n";
        std::cout << "    Cumulative reward: " << std::fixed << std::setprecision(2) << cum_reward << "\n";
    }

    // -- Top divergence rows --
    std::cout << "\n";
    std::cout << "  TRANSITION MODEL DIVERGENCE\n";
    std::cout << "  ---------------------------\n";
    int K = std::min<int>(rows.size(), 3);
    for (int i = 0; i < K; ++i) {
        const auto& r = rows[i];
        std::cout << "  Rank " << (i + 1);
        if (i == 0) std::cout << " (PRIMARY SUSPECT)";
        std::cout << ":\n";
        std::cout << "    (state, action):    (" << r.state << ", " << r.action << ")\n";
        std::cout << "    KL divergence:      " << std::fixed << std::setprecision(4)
                  << r.kl_divergence << "\n";
        std::cout << "    Observations:       " << r.observation_count;
        if (r.observation_count < 30) std::cout << "  [LOW_CONFIDENCE]";
        std::cout << "\n";
        if (!r.worst_dest.empty()) {
            std::cout << "    Worst dest mismatch on '" << r.worst_dest << "':\n";
            std::cout << "      Model assumed:    " << std::setprecision(4) << r.worst_model_p << "\n";
            std::cout << "      Empirical:        " << r.worst_empirical_p << "\n";
            std::cout << "      Underestimation factor: " << std::setprecision(2)
                      << r.worst_underestimation_factor << "x\n";
        }
        std::cout << "\n";
    }
    if (rows.empty()) {
        std::cout << "    (no (s, a) pairs observed in the log)\n\n";
    }

    // -- Backwards solver --
    if (solver) {
        std::cout << "  MINIMUM MODEL EDIT TO TRIGGER VERIFY FAILURE\n";
        std::cout << "  --------------------------------------------\n";
        if (!solver->found && solver->alpha >= 1.0) {
            std::cout << "    No perturbation along the empirical direction (up to alpha=1.0,\n";
            std::cout << "    i.e. believing the empirical distribution entirely) breaks any\n";
            std::cout << "    VERIFY assertion. The model is robust to the observed reality\n";
            std::cout << "    -- the failure is not explained by transition mis-specification\n";
            std::cout << "    captured in any VERIFY block.\n\n";
        } else if (solver->alpha <= 1e-9) {
            std::cout << "    The model already fails VERIFY at alpha=0 (unperturbed). Run\n";
            std::cout << "    the compiler's own --diagnose first.\n\n";
        } else {
            std::cout << "    Minimum perturbation: ||DeltaT||_1 = " << std::fixed
                      << std::setprecision(4) << solver->l1_norm << "\n";
            std::cout << "    (alpha = " << solver->alpha
                      << " along the empirical direction)\n";
            std::cout << "    Triggering assertion: " << solver->triggering_assertion << "\n";

            // Top 3 (s,a,s') deltas by |Δ|.
            auto deltas = solver->deltas;
            std::sort(deltas.begin(), deltas.end(),
                      [](const auto& a, const auto& b) {
                          return std::abs(std::get<3>(a)) > std::abs(std::get<3>(b));
                      });
            int dK = std::min<int>(deltas.size(), 3);
            if (dK > 0) {
                std::cout << "    Largest individual edits:\n";
                for (int i = 0; i < dK; ++i) {
                    const auto& [s, a, sp, d] = deltas[i];
                    std::cout << "      T(" << s << ", " << a << ", " << sp << "): "
                              << (d >= 0 ? "+" : "") << std::fixed << std::setprecision(4)
                              << d << "\n";
                }
            }
            std::cout << "\n";
        }
    }

    // -- Verdict --
    std::cout << "  VERDICT\n";
    std::cout << "  -------\n";
    std::cout << "    [" << verdictName(verdict) << "]";
    switch (verdict) {
        case Verdict::ACCURATE:
            std::cout << "  All empirical transitions are within 2x of the model.\n";
            break;
        case Verdict::OPTIMISTIC:
            std::cout << "  Primary suspect underestimated by 2-10x.\n";
            break;
        case Verdict::DANGEROUS:
            std::cout << "  Primary suspect underestimated by 10x or more.\n";
            break;
    }
    std::cout << "======================================\n";
}

// -- JSON emitter --

static std::string jsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += " ";
                else out.push_back(c);
        }
    }
    return out;
}

static void printJsonReport(const std::string& model_path,
                            const std::string& log_path,
                            const ExecutionLog& log,
                            const std::vector<DivergenceRow>& rows,
                            const BackwardsSolverResult* solver,
                            Verdict verdict)
{
    std::ostringstream j;
    j << "{\n";
    j << "  \"version\": \"3.0\",\n";
    j << "  \"model\": \"" << jsonEscape(model_path) << "\",\n";
    j << "  \"log\": \""   << jsonEscape(log_path)   << "\",\n";
    j << "  \"log_steps\": " << log.steps.size() << ",\n";
    j << "  \"log_failed\": " << (log.failed ? "true" : "false") << ",\n";

    // Divergence rows
    j << "  \"divergence\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        j << "    {";
        j << "\"state\": \""     << jsonEscape(r.state)  << "\", ";
        j << "\"action\": \""    << jsonEscape(r.action) << "\", ";
        j << "\"kl_divergence\": " << r.kl_divergence    << ", ";
        j << "\"obs_count\": "     << r.observation_count << ", ";
        j << "\"worst_dest\": \"" << jsonEscape(r.worst_dest) << "\", ";
        j << "\"worst_model_p\": " << r.worst_model_p      << ", ";
        j << "\"worst_empirical_p\": " << r.worst_empirical_p << ", ";
        j << "\"underestimation_factor\": " << r.worst_underestimation_factor;
        j << "}";
        if (i + 1 < rows.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";

    // Backwards solver
    if (solver) {
        j << "  \"backwards_solver\": {\n";
        j << "    \"found\": " << (solver->found ? "true" : "false") << ",\n";
        j << "    \"alpha\": " << solver->alpha << ",\n";
        j << "    \"l1_norm\": " << solver->l1_norm << ",\n";
        j << "    \"triggering_assertion\": \""
          << jsonEscape(solver->triggering_assertion) << "\",\n";
        j << "    \"deltas\": [";
        for (size_t i = 0; i < solver->deltas.size(); ++i) {
            const auto& [s, a, sp, d] = solver->deltas[i];
            j << "[\"" << jsonEscape(s) << "\",\"" << jsonEscape(a)
              << "\",\"" << jsonEscape(sp) << "\"," << d << "]";
            if (i + 1 < solver->deltas.size()) j << ",";
        }
        j << "]\n";
        j << "  },\n";
    } else {
        j << "  \"backwards_solver\": null,\n";
    }

    j << "  \"verdict\": \"" << verdictName(verdict) << "\",\n";
    j << "  \"verdict_exit_code\": " << static_cast<int>(verdict) << "\n";
    j << "}\n";
    std::cout << j.str();
}

// ============================================================================
// SECTION F: MAIN
// ============================================================================

#ifndef MDP_AUTOPSY_NO_MAIN

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <model.mdp> <run.log> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --report-only    Skip the backwards solver (only divergence report).\n"
              << "  --json           Emit machine-readable JSON to stdout.\n"
              << "\n"
              << "Exit codes:\n"
              << "  0 = MODEL_ACCURATE   (all empirical transitions within 2x of model)\n"
              << "  1 = MODEL_OPTIMISTIC (primary suspect underestimated by 2-10x)\n"
              << "  2 = MODEL_DANGEROUS  (primary suspect underestimated by 10x or more)\n"
              << "  3 = parse error\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return 3; }

    std::string model_path = argv[1];
    std::string log_path   = argv[2];
    bool flag_report_only  = false;
    bool flag_json         = false;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--report-only") flag_report_only = true;
        else if (arg == "--json")        flag_json        = true;
        else {
            std::cerr << "[mdp_autopsy] Unknown flag: " << arg << "\n";
            printUsage(argv[0]);
            return 3;
        }
    }

    // ---- Parse the model ----
    std::ifstream mf(model_path);
    if (!mf) {
        std::cerr << "[mdp_autopsy] Cannot open model: " << model_path << "\n";
        return 3;
    }
    std::stringstream buf; buf << mf.rdbuf();
    MDP_AST ast;
    ValidationResult vr;
    {
        // Suppress parser chatter via stream redirect.
        std::stringstream sink;
        std::streambuf* orig = std::cout.rdbuf(sink.rdbuf());
        ast = parseMDPString(buf.str());

        // Phase 2A: if the model declares BRIDGE: hmm -> mdp, run the bridge
        // exactly as the compiler does -- this populates HMM states as MDP
        // STATEs and adds their transition rows. Without this, models that
        // rely on HMM_STATES alone fail validation in the autopsy.
        if (ast.hmm_bridge_enabled) {
            auto bridge_errors = bridgeHMMToMDP(ast);
            std::cout.rdbuf(orig);
            if (!bridge_errors.empty()) {
                std::cerr << "[mdp_autopsy] HMM bridge failed:\n";
                for (const auto& e : bridge_errors) std::cerr << "  " << e << "\n";
                return 3;
            }
            orig = std::cout.rdbuf(sink.rdbuf());
        }

        vr  = validateAST(ast);
        std::cout.rdbuf(orig);
    }
    if (!vr.errors.empty()) {
        std::cerr << "[mdp_autopsy] Model failed validation:\n";
        for (const auto& e : vr.errors) std::cerr << "  " << e << "\n";
        return 3;
    }

    // ---- Parse the log ----
    ExecutionLog log;
    std::string err;
    if (!parseLogFile(log_path, log, err)) {
        std::cerr << "[mdp_autopsy] " << err << "\n";
        return 3;
    }
    if (!validateLogAgainstAST(log, ast, err)) {
        std::cerr << "[mdp_autopsy] " << err << "\n";
        return 3;
    }

    // ---- Empirical analysis ----
    EmpiricalTransitions emp = buildEmpirical(log, &ast);
    std::vector<DivergenceRow> rows = rankByKL(ast, emp);

    BackwardsSolverResult solver{};
    bool ran_solver = false;
    if (!flag_report_only && !ast.verify_assertions.empty()) {
        solver = runBackwardsSolver(ast, emp);
        ran_solver = true;
    }

    Verdict verdict = classifyVerdict(rows);

    if (flag_json) {
        printJsonReport(model_path, log_path, log, rows,
                        ran_solver ? &solver : nullptr, verdict);
    } else {
        printHumanReport(model_path, log_path, log, rows,
                         ran_solver ? &solver : nullptr, verdict);
    }
    return static_cast<int>(verdict);
}

#endif  // MDP_AUTOPSY_NO_MAIN
