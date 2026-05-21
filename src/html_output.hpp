// ============================================================================
// FILE:    html_output.hpp
// PURPOSE: Generate a self-contained HTML file visualizing ALL solver results.
//          Shows VI grid, PI grid, Q-Learning grid side by side with tabs.
//          v3.0: also renders a POMDP panel when PBVIResult is provided —
//          belief simplex, alpha-vector envelope, named-belief table.
// ============================================================================
#ifndef HTML_OUTPUT_HPP_INCLUDED
#define HTML_OUTPUT_HPP_INCLUDED

#include <string>

struct MDP_AST;
struct SolverResult;
struct PolicyIterationResult;
struct QLearningResult;
struct PBVIResult;  // v3.0 — Point-Based Value Iteration result

// Generate HTML with all available solver results.
// vi_result and pi_result may be nullptr if those solvers were not run.
// ql_result may be nullptr if Q-Learning was not run.
// pbvi_result may be nullptr (v3.0); when non-null, an extra POMDP panel
// renders the belief simplex and alpha-vector envelope.
bool generateHTML(const std::string& output_path,
                  const MDP_AST& ast,
                  const SolverResult* vi_result,
                  const PolicyIterationResult* pi_result,
                  const QLearningResult* ql_result,
                  const PBVIResult* pbvi_result = nullptr);

#endif
