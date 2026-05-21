// ============================================================================
// FILE:    html_output.cpp
// PURPOSE: Generate a single self-contained HTML file that visualizes ALL
//          solver results: Value Iteration, Policy Iteration, Q-Learning.
//          Each solver gets its own grid tab showing its policy and values.
//          Q-Learning tab shows the learned Q-derived values and policy.
//          Includes reward curve, transition graph, hover info panel.
//          Pure vanilla JS/CSS — no external libraries.
// ============================================================================

#include "html_output.hpp"

// MDP_AST, SolverResult, PolicyIterationResult, QLearningResult, getReward
// are provided by the main compiler when compiled as a single translation unit.

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <set>
#include <limits>

// ---- Helper: serialize a values map to JS object ----
static std::string serializeValues(const std::unordered_map<std::string, double>& vals,
                                    const std::vector<std::string>& states) {
    std::ostringstream js;
    js << std::fixed << std::setprecision(6) << "{";
    for (size_t i = 0; i < states.size(); i++) {
        double v = vals.count(states[i]) ? vals.at(states[i]) : 0.0;
        js << "\"" << states[i] << "\":" << v;
        if (i + 1 < states.size()) js << ",";
    }
    js << "}";
    return js.str();
}

// ---- Helper: serialize a policy map to JS object ----
static std::string serializePolicy(const std::unordered_map<std::string, std::string>& pol,
                                    const std::vector<std::string>& states) {
    std::ostringstream js;
    js << "{";
    for (size_t i = 0; i < states.size(); i++) {
        std::string a = pol.count(states[i]) ? pol.at(states[i]) : "?";
        js << "\"" << states[i] << "\":\"" << a << "\"";
        if (i + 1 < states.size()) js << ",";
    }
    js << "}";
    return js.str();
}

// ---- buildJSDataBlock: serialize ALL solver data into a JS object ----
static std::string buildJSDataBlock(const MDP_AST& ast,
                                     const SolverResult* vi,
                                     const PolicyIterationResult* pi,
                                     const QLearningResult* ql) {
    std::ostringstream js;
    js << std::fixed << std::setprecision(6);

    std::vector<std::string> sorted_states(ast.states.begin(), ast.states.end());
    std::sort(sorted_states.begin(), sorted_states.end());

    js << "const MDP_DATA = {\n";

    // ---- States ----
    js << "  states: [";
    for (size_t i = 0; i < sorted_states.size(); i++) {
        js << "\"" << sorted_states[i] << "\"";
        if (i + 1 < sorted_states.size()) js << ",";
    }
    js << "],\n";

    // ---- Actions ----
    js << "  actions: [";
    { std::vector<std::string> sa(ast.actions.begin(), ast.actions.end()); std::sort(sa.begin(), sa.end());
      for (size_t i = 0; i < sa.size(); i++) { js << "\"" << sa[i] << "\""; if (i+1<sa.size()) js << ","; } }
    js << "],\n";

    // ---- Rewards R(s) ----
    js << "  rewards: " << serializeValues(ast.rewards, sorted_states) << ",\n";

    // ---- Transitions ----
    js << "  transitions: {\n";
    std::unordered_map<std::string, std::vector<std::tuple<std::string,std::string,double>>> tmap;
    for (const auto& t : ast.transitions)
        tmap[t.source_state].emplace_back(t.action, t.dest_state, t.probability);
    for (size_t i = 0; i < sorted_states.size(); i++) {
        const auto& s = sorted_states[i];
        js << "    \"" << s << "\":[";
        if (tmap.count(s)) { const auto& ts = tmap.at(s);
            for (size_t j = 0; j < ts.size(); j++) { auto& [a,d,p] = ts[j];
                js << "{a:\"" << a << "\",d:\"" << d << "\",p:" << p << "}";
                if (j+1<ts.size()) js << ","; } }
        js << "]"; if (i+1<sorted_states.size()) js << ","; js << "\n";
    }
    js << "  },\n";

    // ---- Grid detection ----
    std::regex rc_pat("R(\\d+)C(\\d+)"); std::smatch m;
    bool is_grid = true; int max_r=0, max_c=0;
    for (const auto& s : ast.states) {
        if (!std::regex_match(s, m, rc_pat)) { is_grid = false; break; }
        max_r = std::max(max_r, std::stoi(m[1].str()));
        max_c = std::max(max_c, std::stoi(m[2].str()));
    }
    if (ast.states.size() < 2) is_grid = false;
    js << "  isGrid:" << (is_grid?"true":"false") << ", gridRows:" << max_r << ", gridCols:" << max_c << ",\n";

    // ---- Goal/Trap ----
    double maxR=-1e18, minR=1e18; std::string goal_st, trap_st;
    for (const auto& [s,r] : ast.rewards) { if(r>maxR){maxR=r;goal_st=s;} if(r<minR){minR=r;trap_st=s;} }
    js << "  goalState:\"" << goal_st << "\", trapState:\"" << trap_st << "\",\n";
    js << "  gamma:" << ast.discount_factor << ",\n";

    // ---- Solver availability flags ----
    js << "  hasVI:" << (vi?"true":"false") << ",\n";
    js << "  hasPI:" << (pi?"true":"false") << ",\n";
    js << "  hasQL:" << (ql?"true":"false") << ",\n";

    // ==== VALUE ITERATION DATA ====
    if (vi) {
        js << "  vi: {\n";
        js << "    values:" << serializeValues(vi->values, sorted_states) << ",\n";
        js << "    policy:" << serializePolicy(vi->policy, sorted_states) << ",\n";
        js << "    iterations:" << vi->iterations << ",\n";
        js << "    converged:" << (vi->converged?"true":"false") << ",\n";
        js << "    timeMs:" << std::fixed << std::setprecision(2) << vi->wall_clock_ms << "\n";
        js << "  },\n";
    }

    // ==== POLICY ITERATION DATA ====
    if (pi) {
        js << "  pi: {\n";
        js << "    values:" << serializeValues(pi->values, sorted_states) << ",\n";
        js << "    policy:" << serializePolicy(pi->policy, sorted_states) << ",\n";
        js << "    steps:" << pi->policy_improvement_steps << ",\n";
        js << "    bellmanEvals:" << pi->total_bellman_evaluations << ",\n";
        js << "    converged:" << (pi->converged?"true":"false") << ",\n";
        js << "    timeMs:" << std::fixed << std::setprecision(2) << pi->wall_clock_ms << "\n";
        js << "  },\n";
    }

    // ==== Q-LEARNING DATA ====
    if (ql) {
        // Derive V(s) = max_a Q(s,a) from the Q-table
        std::unordered_map<std::string, double> ql_values;
        for (const auto& s : sorted_states) {
            double best = -std::numeric_limits<double>::max();
            if (ql->Q_table.count(s)) {
                for (const auto& [a, q] : ql->Q_table.at(s))
                    if (q > best) best = q;
            }
            ql_values[s] = (best > -1e17) ? best : 0.0;
        }

        js << "  ql: {\n";
        js << "    values:" << serializeValues(ql_values, sorted_states) << ",\n";
        js << "    policy:" << serializePolicy(ql->policy, sorted_states) << ",\n";
        js << "    episodes:" << (int)ql->episode_rewards.size() << ",\n";
        js << "    converged:" << (ql->converged?"true":"false") << ",\n";
        js << "    timeMs:" << std::fixed << std::setprecision(2) << ql->wall_clock_ms << ",\n";

        // Q-table for detailed hover info: ql.qtable[state][action] = value
        js << "    qtable: {\n";
        for (size_t i = 0; i < sorted_states.size(); i++) {
            const auto& s = sorted_states[i];
            js << "      \"" << s << "\":{";
            if (ql->Q_table.count(s)) {
                bool first = true;
                for (const auto& [a, q] : ql->Q_table.at(s)) {
                    if (!first) js << ","; first = false;
                    js << "\"" << a << "\":" << std::fixed << std::setprecision(4) << q;
                }
            }
            js << "}"; if (i+1<sorted_states.size()) js << ","; js << "\n";
        }
        js << "    },\n";

        // Episode rewards for the curve
        js << "    episodeRewards:[";
        size_t step = std::max((size_t)1, ql->episode_rewards.size() / 2000);
        for (size_t i = 0; i < ql->episode_rewards.size(); i += step) {
            js << std::fixed << std::setprecision(2) << ql->episode_rewards[i];
            if (i + step < ql->episode_rewards.size()) js << ",";
        }
        js << "]\n";
        js << "  },\n";
    }

    js << "};\n";
    return js.str();
}


// ---- buildPBVIDataBlock: serialize PBVI result for the POMDP panel ----
// Produces a JS variable POMDP_DATA used by the simplex / envelope renderers.
// Only called when pbvi != nullptr.
static std::string buildPBVIDataBlock(const MDP_AST& ast, const PBVIResult* pbvi) {
    std::ostringstream js;
    js << std::fixed << std::setprecision(6);
    if (!pbvi) { js << "const POMDP_DATA = null;\n"; return js.str(); }

    // Deterministic state ordering for the simplex axes.
    std::vector<std::string> sorted_states(ast.states.begin(), ast.states.end());
    std::sort(sorted_states.begin(), sorted_states.end());

    js << "const POMDP_DATA = {\n";
    js << "  states: [";
    for (size_t i = 0; i < sorted_states.size(); i++) {
        js << "\"" << sorted_states[i] << "\"";
        if (i + 1 < sorted_states.size()) js << ",";
    }
    js << "],\n";

    // Alpha vectors: array of {action: ..., values: {s -> v}}.
    js << "  alphaVectors: [\n";
    for (size_t i = 0; i < pbvi->alpha_vectors.size(); i++) {
        const auto& a = pbvi->alpha_vectors[i];
        js << "    {action: \"" << a.action << "\", values: {";
        bool first = true;
        for (const auto& s : sorted_states) {
            if (!first) js << ",";
            double v = a.values.count(s) ? a.values.at(s) : 0.0;
            js << "\"" << s << "\":" << v;
            first = false;
        }
        js << "}}";
        if (i + 1 < pbvi->alpha_vectors.size()) js << ",";
        js << "\n";
    }
    js << "  ],\n";

    // Belief points: array of {name, probs}.
    js << "  beliefs: [\n";
    for (size_t i = 0; i < pbvi->belief_points.size(); i++) {
        const auto& nb = pbvi->belief_points[i];
        js << "    {name: \"" << nb.name << "\", probs: {";
        bool first = true;
        for (const auto& [s, p] : nb.probs) {
            if (!first) js << ",";
            js << "\"" << s << "\":" << p;
            first = false;
        }
        js << "}, value: "
           << (pbvi->values_at_belief.count(nb.name) ? pbvi->values_at_belief.at(nb.name) : 0.0)
           << ", action: \""
           << (pbvi->policy_at_belief.count(nb.name) ? pbvi->policy_at_belief.at(nb.name) : "")
           << "\"}";
        if (i + 1 < pbvi->belief_points.size()) js << ",";
        js << "\n";
    }
    js << "  ],\n";

    // Observations.
    js << "  observations: [";
    {
        std::vector<std::string> obs(ast.observations.begin(), ast.observations.end());
        std::sort(obs.begin(), obs.end());
        for (size_t i = 0; i < obs.size(); i++) {
            js << "\"" << obs[i] << "\"";
            if (i + 1 < obs.size()) js << ",";
        }
    }
    js << "],\n";

    js << "  iterations: " << pbvi->iterations << ",\n";
    js << "  converged: " << (pbvi->converged ? "true" : "false") << ",\n";
    js << "  walltime: " << pbvi->wall_clock_ms << "\n";
    js << "};\n";
    return js.str();
}


// ---- generateHTML ----
bool generateHTML(const std::string& output_path,
                  const MDP_AST& ast,
                  const SolverResult* vi_result,
                  const PolicyIterationResult* pi_result,
                  const QLearningResult* ql_result,
                  const PBVIResult* pbvi_result) {

    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot write HTML file: " << output_path << "\n";
        return false;
    }

    std::string js_data  = buildJSDataBlock(ast, vi_result, pi_result, ql_result);
    std::string js_pomdp = buildPBVIDataBlock(ast, pbvi_result);

    file << R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MDP-DSL Visualization</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,sans-serif;background:#1a1a2e;color:#eee;display:flex;height:100vh}
#sidebar{width:330px;background:#16213e;padding:18px;overflow-y:auto;border-right:2px solid #0f3460;flex-shrink:0}
#main{flex:1;padding:20px;overflow:auto;display:flex;flex-direction:column;align-items:center}
h1{color:#e94560;font-size:1.4em;margin-bottom:10px}
h2{font-size:1.05em;margin:14px 0 6px;color:#53a8b6}
.stat-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1a1a2e;font-size:0.88em}
.stat-label{color:#888}.stat-value{color:#e94560;font-weight:bold}
#hover-info{background:#0f3460;border-radius:8px;padding:12px;margin-top:8px;min-height:100px}
#hover-info h3{color:#e94560;margin-bottom:6px;font-size:1em}
.trans-item{font-size:0.8em;color:#aaa;padding:2px 0}
.trans-prob{color:#53a8b6}
.qtable-row{font-size:0.8em;color:#ccc;padding:1px 0}
.qtable-val{color:#ffd700;font-weight:bold}
#grid-container{position:relative;margin:20px}
.cell{position:absolute;border:2px solid #333;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;transition:border-color 0.2s,transform 0.1s;font-size:0.82em}
.cell:hover{border-color:#fff;transform:scale(1.05);z-index:10}
.cell-name{font-weight:bold;font-size:0.85em}.cell-value{font-size:1.05em;font-weight:bold}.cell-arrow{font-size:1.7em;line-height:1}
.wall-cell{background:#444!important}
#tab-bar{display:flex;gap:0;margin:10px 0}
.tab{padding:10px 20px;background:#0f3460;color:#aaa;border:1px solid #333;cursor:pointer;font-size:0.95em;font-weight:bold;border-bottom:none;border-radius:6px 6px 0 0;transition:all 0.2s}
.tab:hover{color:#eee;background:#1a3a5c}
.tab.active{background:#1a1a2e;color:#e94560;border-color:#53a8b6;border-bottom:2px solid #1a1a2e}
.tab.disabled{opacity:0.3;cursor:not-allowed}
.tab-label{font-size:0.7em;display:block;color:#53a8b6;margin-top:2px}
#btn-bar{display:flex;gap:10px;margin:8px 0}
.btn{padding:7px 14px;background:#0f3460;color:#eee;border:1px solid #53a8b6;border-radius:4px;cursor:pointer;font-size:0.85em}
.btn:hover{background:#53a8b6;color:#1a1a2e}
.btn.disabled{opacity:0.35;cursor:not-allowed}
#reward-canvas{border:1px solid #333;border-radius:4px;margin-top:8px;background:#111}
#graph-canvas{border:1px solid #333;border-radius:4px;background:#111}
.legend{display:flex;gap:12px;margin:6px 0;font-size:0.78em;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-swatch{width:12px;height:12px;border-radius:3px}
.solver-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:bold;margin-left:6px}
</style>
</head>
<body>

<div id="sidebar">
  <h1>MDP-DSL Visualizer</h1>
  <h2>Solver Stats</h2>
  <div id="stats-panel"></div>
  <h2>Cell Info (hover)</h2>
  <div id="hover-info">
    <h3 id="hi-name">Hover over a cell</h3>
    <div id="hi-details"></div>
  </div>
  <h2>Legend</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:hsl(0,70%,45%)"></div>Low V*</div>
    <div class="legend-item"><div class="legend-swatch" style="background:hsl(60,70%,45%)"></div>Mid</div>
    <div class="legend-item"><div class="legend-swatch" style="background:hsl(120,70%,45%)"></div>High V*</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#ffd700"></div>Q-Learning</div>
  </div>
  <h2>Q-Learning Reward Curve</h2>
  <button class="btn" id="btn-ql-curve" onclick="playRewardCurve()">Play Curve</button>
  <canvas id="reward-canvas" width="290" height="140"></canvas>
</div>

<div id="main">
  <div id="tab-bar"></div>
  <div id="btn-bar">
    <button class="btn" onclick="showGraph()">Transition Graph</button>
  </div>
  <div id="grid-container"></div>
  <canvas id="graph-canvas" width="750" height="500" style="display:none"></canvas>
  <!-- v3.0 POMDP panel — hidden unless PBVI ran -->
  <div id="pomdp-panel" style="display:none;width:100%;max-width:880px;">
    <h2 style="color:#53a8b6;margin-top:6px">Belief Simplex &amp; Alpha-Vector Envelope</h2>
    <canvas id="pomdp-simplex" width="820" height="320"
            style="background:#0f1530;border-radius:6px;display:block;margin:8px 0;"></canvas>
    <div id="pomdp-legend" style="font-size:0.85em;color:#aaa;margin-bottom:14px;"></div>

    <h2 style="color:#53a8b6">Alpha-Vectors (upper envelope of V*)</h2>
    <div id="pomdp-alpha-table" style="margin:8px 0 14px 0;"></div>

    <h2 style="color:#53a8b6">Named Beliefs &amp; Initial Belief</h2>
    <div id="pomdp-belief-table" style="margin:8px 0 14px 0;"></div>

    <div id="pomdp-summary" style="font-size:0.85em;color:#888;padding:6px;
                                  background:#0f3460;border-radius:6px;"></div>
  </div>
</div>

<script>
)html";

    file << js_data;
    file << js_pomdp;

    file << R"js(
// ---- Globals ----
let currentSolver = null; // 'vi', 'pi', 'ql', 'pomdp'

function valueToHue(v, vMin, vMax) {
  if (vMax <= vMin) return 60;
  return Math.max(0, Math.min(120, ((v - vMin) / (vMax - vMin)) * 120));
}
function actionArrow(a) {
  const m = {Up:'\u2191',Down:'\u2193',Left:'\u2190',Right:'\u2192',
             MoveRight:'\u2192',MoveLeft:'\u2190',Stay:'\u25CF','(terminal)':'\u25CF'};
  return m[a] || a.charAt(0);
}

// ---- Build solver tabs ----
function buildTabs() {
  const bar = document.getElementById('tab-bar');
  bar.innerHTML = '';
  const solvers = [];
  if (MDP_DATA.hasVI) solvers.push({key:'vi', name:'Value Iteration', short:'VI'});
  if (MDP_DATA.hasPI) solvers.push({key:'pi', name:'Policy Iteration', short:'PI'});
  if (MDP_DATA.hasQL) solvers.push({key:'ql', name:'Q-Learning', short:'QL'});
  if (typeof POMDP_DATA !== 'undefined' && POMDP_DATA)
      solvers.push({key:'pomdp', name:'PBVI (POMDP)', short:'POMDP'});

  solvers.forEach(s => {
    const tab = document.createElement('div');
    tab.className = 'tab';
    tab.id = 'tab-' + s.key;
    tab.innerHTML = s.short + '<span class="tab-label">' + s.name + '</span>';
    tab.onclick = () => switchSolver(s.key);
    bar.appendChild(tab);
  });

  // Default to first available
  if (solvers.length > 0) switchSolver(solvers[0].key);
}

// ---- Get solver data by key ----
function getSolverData(key) {
  if (key === 'vi' && MDP_DATA.hasVI) return MDP_DATA.vi;
  if (key === 'pi' && MDP_DATA.hasPI) return MDP_DATA.pi;
  if (key === 'ql' && MDP_DATA.hasQL) return MDP_DATA.ql;
  return null;
}

// ---- Switch active solver tab ----
function switchSolver(key) {
  currentSolver = key;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById('tab-' + key);
  if (tab) tab.classList.add('active');

  // POMDP mode hides the grid and shows the simplex/envelope panel.
  if (key === 'pomdp') {
    document.getElementById('graph-canvas').style.display = 'none';
    document.getElementById('grid-container').style.display = 'none';
    document.getElementById('pomdp-panel').style.display = 'block';
    renderStatsPOMDP();
    renderPOMDP();
    const qlBtn = document.getElementById('btn-ql-curve');
    if (qlBtn) qlBtn.classList.add('disabled');
    return;
  }

  // Else: MDP solver mode.
  document.getElementById('pomdp-panel').style.display = 'none';
  renderStats(key);
  document.getElementById('graph-canvas').style.display = 'none';
  document.getElementById('grid-container').style.display = 'block';
  showGrid(key);

  const qlBtn = document.getElementById('btn-ql-curve');
  if (MDP_DATA.hasQL && MDP_DATA.ql.episodeRewards && MDP_DATA.ql.episodeRewards.length > 0) {
    qlBtn.classList.remove('disabled');
  } else {
    qlBtn.classList.add('disabled');
  }
}

// ---- POMDP stats ----
function renderStatsPOMDP() {
  const panel = document.getElementById('stats-panel');
  if (!panel) return;
  const d = POMDP_DATA;
  const rows = [
    ['Solver', 'PBVI'],
    ['Alpha vectors', d.alphaVectors.length],
    ['Belief points', d.beliefs.length],
    ['Iterations', d.iterations],
    ['Converged', d.converged ? 'Yes' : 'No'],
    ['Wall time (ms)', (d.walltime || 0).toFixed(1)],
    ['States', d.states.length],
    ['Observations', d.observations.length],
  ];
  panel.innerHTML = rows.map(r =>
    '<div class="stat-row"><span class="stat-label">' + r[0] +
    '</span><span class="stat-value">' + r[1] + '</span></div>'
  ).join('');
}

// ---- POMDP panel renderer ----
// For 2-state POMDPs, draws the alpha-vector envelope as a function of
// b(states[0]) and color-codes the active-action regions. For larger state
// spaces, draws an alpha-vector table only.
function renderPOMDP() {
  const d = POMDP_DATA;
  const cv = document.getElementById('pomdp-simplex');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#0f1530'; ctx.fillRect(0,0,W,H);

  const padL = 50, padR = 20, padT = 20, padB = 50;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Distinct color per action (deterministic by action label).
  const actionColors = {};
  const palette = ['#e94560','#f2c94c','#21bf73','#56cbf9','#a37afc','#ffaaaa','#88dd88','#ffcc77'];
  let ci = 0;
  function colorFor(a){ if (!(a in actionColors)) actionColors[a] = palette[ci++ % palette.length]; return actionColors[a]; }

  if (d.states.length === 2) {
    // ---- 2-state simplex: 1D belief axis ----
    const s0 = d.states[0], s1 = d.states[1];

    // Sample V(b) and best-action a*(b) over a grid of b values.
    const N = 401;
    const bs = [], vs = [], acts = [];
    let vMin = Infinity, vMax = -Infinity;
    for (let i = 0; i < N; i++) {
      const b = i / (N-1);
      let best = -Infinity, bestA = '?';
      for (const av of d.alphaVectors) {
        const val = av.values[s0]*b + av.values[s1]*(1-b);
        if (val > best) { best = val; bestA = av.action; }
      }
      bs.push(b); vs.push(best); acts.push(bestA);
      if (best < vMin) vMin = best;
      if (best > vMax) vMax = best;
    }
    if (!isFinite(vMin) || !isFinite(vMax) || vMax === vMin) { vMin -= 1; vMax += 1; }

    // Draw colored action-region bands along the b axis (bottom strip).
    const stripH = 18;
    const stripY = padT + plotH + 6;
    for (let i = 0; i < N-1; i++) {
      const x0 = padL + bs[i]   * plotW;
      const x1 = padL + bs[i+1] * plotW;
      ctx.fillStyle = colorFor(acts[i]);
      ctx.fillRect(x0, stripY, x1-x0+1, stripH);
    }

    // Draw value curve V*(b) over the plot area.
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = padL + bs[i] * plotW;
      const y = padT + plotH - (vs[i] - vMin) / (vMax - vMin) * plotH;
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();

    // Draw each alpha vector as its own faint line (the "envelope" decomposition).
    ctx.lineWidth = 1;
    for (const av of d.alphaVectors) {
      ctx.strokeStyle = colorFor(av.action) + 'aa';
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const b = bs[i];
        const val = av.values[s0]*b + av.values[s1]*(1-b);
        const x = padL + b * plotW;
        const y = padT + plotH - (val - vMin) / (vMax - vMin) * plotH;
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }

    // Axes.
    ctx.strokeStyle = '#888'; ctx.lineWidth = 1; ctx.font='11px sans-serif';
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+plotH); ctx.lineTo(padL+plotW, padT+plotH);
    ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.textAlign = 'center';
    ctx.fillText('b('+s0+')',  padL + plotW/2, H - 8);
    ctx.fillText('0.0', padL, padT+plotH+14);
    ctx.fillText('0.5', padL + plotW/2, padT+plotH+14);
    ctx.fillText('1.0', padL + plotW, padT+plotH+14);
    ctx.textAlign='right';
    ctx.fillText('V*='+vMax.toFixed(2), padL-4, padT+10);
    ctx.fillText('V*='+vMin.toFixed(2), padL-4, padT+plotH);
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ccc';
    ctx.fillText('Action region', padL + plotW + 6, stripY + 14);

    // Mark each named belief on the b axis.
    ctx.font='10px sans-serif';
    for (const bp of d.beliefs) {
      // Skip auto-generated anonymous belief points (names starting with _b)
      // to keep the chart readable.
      if (/^_b/.test(bp.name)) continue;
      const bv = bp.probs[s0] || 0;
      const x = padL + bv * plotW;
      const y = padT + plotH - (bp.value - vMin) / (vMax - vMin) * plotH;
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#ffd700';
      ctx.fillText(bp.name, x + 6, y - 4);
    }

    // Build the legend (action colors).
    const legend = document.getElementById('pomdp-legend');
    legend.innerHTML = 'Bottom strip: optimal action region as a function of b('+s0+'). '+
       'Gold curve: V*(b). Faint lines: individual α-vectors. <br>'+
       Object.keys(actionColors).map(a =>
         '<span style="display:inline-block;width:11px;height:11px;background:'+actionColors[a]+
         ';margin:0 4px -1px 8px;border-radius:2px"></span>'+a).join('');

  } else {
    // ≥3 states: draw alpha-vector bars per state instead of a simplex curve.
    ctx.fillStyle = '#aaa';
    ctx.textAlign='center';
    ctx.fillText('Belief simplex visualisation supported for 2-state POMDPs only.',
                 W/2, H/2 - 6);
    ctx.fillText('See the alpha-vector table below for full envelope data.',
                 W/2, H/2 + 12);
    document.getElementById('pomdp-legend').textContent = '';
  }

  // Alpha-vector table.
  let html = '<table style="width:100%;border-collapse:collapse;font-size:0.85em;">';
  html += '<tr style="background:#0f3460;"><th style="text-align:left;padding:4px;">#</th>'+
          '<th style="text-align:left;padding:4px;">action</th>';
  for (const s of d.states) html += '<th style="text-align:right;padding:4px;">α('+s+')</th>';
  html += '</tr>';
  for (let i = 0; i < d.alphaVectors.length; i++) {
    const av = d.alphaVectors[i];
    html += '<tr style="border-bottom:1px solid #233;">';
    html += '<td style="padding:4px;color:#888">'+(i+1)+'</td>';
    html += '<td style="padding:4px;color:'+colorFor(av.action)+';font-weight:bold;">'+av.action+'</td>';
    for (const s of d.states) {
      const v = av.values[s] || 0;
      html += '<td style="padding:4px;text-align:right;color:#ccc;">'+v.toFixed(3)+'</td>';
    }
    html += '</tr>';
  }
  html += '</table>';
  document.getElementById('pomdp-alpha-table').innerHTML = html;

  // Belief table.
  let bhtml = '<table style="width:100%;border-collapse:collapse;font-size:0.85em;">';
  bhtml += '<tr style="background:#0f3460;"><th style="text-align:left;padding:4px;">name</th>';
  for (const s of d.states) bhtml += '<th style="text-align:right;padding:4px;">b('+s+')</th>';
  bhtml += '<th style="text-align:right;padding:4px;">V*(b)</th>'+
           '<th style="text-align:left;padding:4px;">π*(b)</th></tr>';
  for (const bp of d.beliefs) {
    if (/^_b/.test(bp.name)) continue;  // skip auto-expanded points in the user view
    bhtml += '<tr style="border-bottom:1px solid #233;">';
    bhtml += '<td style="padding:4px;color:#ffd700;font-weight:bold">'+bp.name+'</td>';
    for (const s of d.states) {
      const p = bp.probs[s] || 0;
      bhtml += '<td style="padding:4px;text-align:right;color:#ccc;">'+p.toFixed(3)+'</td>';
    }
    bhtml += '<td style="padding:4px;text-align:right;color:#56cbf9;">'+(bp.value||0).toFixed(3)+'</td>';
    bhtml += '<td style="padding:4px;color:'+colorFor(bp.action)+';font-weight:bold;">'+bp.action+'</td>';
    bhtml += '</tr>';
  }
  bhtml += '</table>';
  document.getElementById('pomdp-belief-table').innerHTML = bhtml;

  // Summary.
  document.getElementById('pomdp-summary').innerHTML =
    'PBVI ran <b>'+d.iterations+'</b> backups and retained '+
    '<b>'+d.alphaVectors.length+'</b> α-vectors over a belief set of <b>'+
    d.beliefs.length+'</b> points. Convergence: <b>'+(d.converged?'yes':'no')+
    '</b>. The optimal policy is the upper envelope of these α-vectors over the belief simplex.';
}

// ---- Render stats for a specific solver ----
function renderStats(key) {
  const panel = document.getElementById('stats-panel');
  const d = getSolverData(key);
  if (!d) { panel.innerHTML = '<div class="stat-row"><span class="stat-label">No data</span></div>'; return; }

  let rows = [];
  rows.push(['Solver', {vi:'Value Iteration',pi:'Policy Iteration',ql:'Q-Learning'}[key]]);
  rows.push(['Discount (gamma)', MDP_DATA.gamma.toFixed(2)]);
  rows.push(['States', MDP_DATA.states.length]);

  if (key === 'vi') {
    rows.push(['Iterations', d.iterations]);
    rows.push(['Converged', d.converged ? 'Yes' : 'No']);
    rows.push(['Time', d.timeMs.toFixed(2) + ' ms']);
  } else if (key === 'pi') {
    rows.push(['Policy Steps', d.steps]);
    rows.push(['Bellman Evals', d.bellmanEvals]);
    rows.push(['Converged', d.converged ? 'Yes' : 'No']);
    rows.push(['Time', d.timeMs.toFixed(2) + ' ms']);
  } else if (key === 'ql') {
    rows.push(['Episodes', d.episodes]);
    rows.push(['Converged', d.converged ? 'Yes' : 'Approx.']);
    rows.push(['Time', d.timeMs.toFixed(2) + ' ms']);
  }

  // If VI is available, show policy match for PI/QL
  if (key !== 'vi' && MDP_DATA.hasVI) {
    let match = 0;
    MDP_DATA.states.forEach(s => {
      if (MDP_DATA.vi.policy[s] === d.policy[s]) match++;
    });
    rows.push(['Policy Match vs VI', match + '/' + MDP_DATA.states.length]);
  }

  panel.innerHTML = rows.map(r =>
    '<div class="stat-row"><span class="stat-label">' + r[0] +
    '</span><span class="stat-value">' + r[1] + '</span></div>'
  ).join('');
}

// ---- Draw grid for a specific solver ----
function showGrid(key) {
  document.getElementById('grid-container').style.display = 'block';
  document.getElementById('graph-canvas').style.display = 'none';
  const container = document.getElementById('grid-container');
  container.innerHTML = '';

  const d = getSolverData(key);
  if (!d) return;

  const values = d.values;
  const policy = d.policy;

  // Compute V range for this solver
  const vals = Object.values(values);
  const vMin = Math.min(...vals);
  const vMax = Math.max(...vals);

  const CELL_W = 120, CELL_H = 95, GAP = 4;

  // Solver color accent
  const accents = {vi: '#53a8b6', pi: '#9b59b6', ql: '#ffd700'};
  const accent = accents[key] || '#53a8b6';

  if (MDP_DATA.isGrid) {
    const rows = MDP_DATA.gridRows, cols = MDP_DATA.gridCols;
    container.style.width = (cols * (CELL_W + GAP)) + 'px';
    container.style.height = (rows * (CELL_H + GAP)) + 'px';
    container.style.position = 'relative';
    const stateSet = new Set(MDP_DATA.states);

    for (let r = rows; r >= 1; r--) {
      for (let c = 1; c <= cols; c++) {
        const name = 'R' + r + 'C' + c;
        const x = (c - 1) * (CELL_W + GAP);
        const y = (rows - r) * (CELL_H + GAP);
        const div = document.createElement('div');
        div.className = 'cell';
        div.style.left = x + 'px'; div.style.top = y + 'px';
        div.style.width = CELL_W + 'px'; div.style.height = CELL_H + 'px';
        div.style.borderColor = '#333';

        if (!stateSet.has(name)) {
          div.classList.add('wall-cell');
          div.innerHTML = '<span style="color:#888">WALL</span>';
        } else {
          const v = values[name] || 0;
          const hue = valueToHue(v, vMin, vMax);
          let bg = 'hsl(' + hue + ',70%,35%)';
          if (name === MDP_DATA.goalState) bg = 'hsl(120,80%,40%)';
          if (name === MDP_DATA.trapState) bg = 'hsl(0,80%,35%)';
          div.style.background = bg;
          div.style.color = '#fff';

          const arrow = actionArrow(policy[name] || '?');
          const solverTag = '<span style="position:absolute;top:3px;right:5px;font-size:0.6em;color:' +
                            accent + ';font-weight:bold">' + key.toUpperCase() + '</span>';
          div.innerHTML = solverTag +
            '<div class="cell-name">' + name + '</div>' +
            '<div class="cell-value">' + v.toFixed(2) + '</div>' +
            '<div class="cell-arrow">' + arrow + '</div>';

          div.addEventListener('mouseenter', () => showHoverInfo(name, key));
        }
        container.appendChild(div);
      }
    }
  } else {
    // Non-grid layout
    container.style.display = 'flex';
    container.style.flexWrap = 'wrap';
    container.style.gap = '8px';
    container.style.position = 'relative';
    MDP_DATA.states.forEach(name => {
      const div = document.createElement('div');
      div.style.width = CELL_W + 'px'; div.style.height = CELL_H + 'px';
      div.style.position = 'relative'; div.className = 'cell';
      const v = values[name] || 0;
      const hue = valueToHue(v, vMin, vMax);
      div.style.background = 'hsl(' + hue + ',70%,35%)';
      div.style.color = '#fff';
      const arrow = actionArrow(policy[name] || '?');
      div.innerHTML = '<div class="cell-name">' + name + '</div>' +
        '<div class="cell-value">' + v.toFixed(2) + '</div>' +
        '<div class="cell-arrow">' + arrow + '</div>';
      div.addEventListener('mouseenter', () => showHoverInfo(name, key));
      container.appendChild(div);
    });
  }
}

// ---- Hover info: shows V*, R, policy, transitions, Q-values if QL ----
function showHoverInfo(name, solverKey) {
  document.getElementById('hi-name').textContent = name;
  const d = getSolverData(solverKey);
  if (!d) return;

  const v = (d.values[name] || 0).toFixed(4);
  const r = (MDP_DATA.rewards[name] || 0).toFixed(4);
  const a = d.policy[name] || '?';
  const trans = MDP_DATA.transitions[name] || [];

  let html = '';
  html += '<div class="stat-row"><span class="stat-label">Solver</span><span class="stat-value">' +
          {vi:'Value Iteration',pi:'Policy Iteration',ql:'Q-Learning'}[solverKey] + '</span></div>';
  html += '<div class="stat-row"><span class="stat-label">V*(s)</span><span class="stat-value">' + v + '</span></div>';
  html += '<div class="stat-row"><span class="stat-label">R(s)</span><span class="stat-value">' + r + '</span></div>';
  html += '<div class="stat-row"><span class="stat-label">pi*(s)</span><span class="stat-value" style="color:#e94560">' + a + ' ' + actionArrow(a) + '</span></div>';

  // Show Q-values if this is Q-Learning
  if (solverKey === 'ql' && MDP_DATA.ql && MDP_DATA.ql.qtable && MDP_DATA.ql.qtable[name]) {
    html += '<h3 style="margin-top:8px;font-size:0.9em;color:#ffd700">Q-Values</h3>';
    const qt = MDP_DATA.ql.qtable[name];
    for (const act of Object.keys(qt).sort()) {
      const isOptimal = (act === a);
      const style = isOptimal ? 'color:#ffd700;font-weight:bold' : 'color:#aaa';
      html += '<div class="qtable-row" style="' + style + '">' +
              'Q(' + name + ',' + act + ') = <span class="qtable-val">' + qt[act].toFixed(4) + '</span>' +
              (isOptimal ? ' <-- optimal' : '') + '</div>';
    }
  }

  // Show VI value for comparison when viewing PI or QL
  if (solverKey !== 'vi' && MDP_DATA.hasVI) {
    const viV = (MDP_DATA.vi.values[name] || 0).toFixed(4);
    const viA = MDP_DATA.vi.policy[name] || '?';
    html += '<h3 style="margin-top:8px;font-size:0.85em;color:#53a8b6">VI Reference</h3>';
    html += '<div class="stat-row"><span class="stat-label">VI V*(s)</span><span class="stat-value" style="color:#53a8b6">' + viV + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">VI pi*(s)</span><span class="stat-value" style="color:#53a8b6">' + viA + '</span></div>';
    const match = (viA === a);
    html += '<div class="stat-row"><span class="stat-label">Policy Match</span><span class="stat-value" style="color:' + (match?'#27ae60':'#e94560') + '">' + (match?'YES':'DIFFER') + '</span></div>';
  }

  html += '<h3 style="margin-top:8px;font-size:0.85em">Transitions</h3>';
  trans.forEach(t => {
    html += '<div class="trans-item">' + t.a + ': ' + name + ' &rarr; ' + t.d +
            ' <span class="trans-prob">(p=' + t.p.toFixed(2) + ')</span></div>';
  });

  document.getElementById('hi-details').innerHTML = html;
}

// ---- Q-Learning reward curve ----
function playRewardCurve() {
  if (!MDP_DATA.hasQL || !MDP_DATA.ql.episodeRewards || MDP_DATA.ql.episodeRewards.length === 0) return;
  const canvas = document.getElementById('reward-canvas');
  const ctx = canvas.getContext('2d');
  const data = MDP_DATA.ql.episodeRewards;
  const n = data.length, w = canvas.width, h = canvas.height;
  const mn = Math.min(...data), mx = Math.max(...data);
  const range = mx - mn || 1;
  let frame = 0;
  const step = Math.max(1, Math.floor(n / w));
  function draw() {
    ctx.fillStyle = '#111'; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i <= frame && i < n; i += step) {
      const x = (i/n)*w, y = h - ((data[i]-mn)/range)*(h-10) - 5;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = '#888'; ctx.font = '10px sans-serif';
    ctx.fillText('Episode', w-50, h-3); ctx.fillText('Reward', 2, 12);
    frame += step * 3;
    if (frame < n) requestAnimationFrame(draw);
  }
  frame = 0; draw();
}

// ---- Force-directed graph (uses current solver's policy for highlights) ----
function showGraph() {
  document.getElementById('grid-container').style.display = 'none';
  const canvas = document.getElementById('graph-canvas');
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const d = getSolverData(currentSolver);
  const pol = d ? d.policy : {};
  const values = d ? d.values : {};
  const vals = Object.values(values);
  const vMin = Math.min(...(vals.length?vals:[0]));
  const vMax = Math.max(...(vals.length?vals:[0]));

  const nodes = {};
  MDP_DATA.states.forEach(s => { nodes[s] = {x:W/2+(Math.random()-0.5)*W*0.6, y:H/2+(Math.random()-0.5)*H*0.6, vx:0, vy:0}; });
  const edgeMap = {};
  for (const src of MDP_DATA.states) {
    (MDP_DATA.transitions[src]||[]).forEach(t => {
      if (t.d===src) return;
      const key = src+'>'+t.d;
      if (!edgeMap[key]) edgeMap[key] = {src,dst:t.d,prob:0,isPolicy:false};
      edgeMap[key].prob += t.p;
      if (pol[src]===t.a) edgeMap[key].isPolicy = true;
    });
  }
  const edges = Object.values(edgeMap);
  const K_SPRING=0.005,K_REPEL=8000,DAMP=0.85;

  function simulate() {
    const sl = MDP_DATA.states;
    for(let i=0;i<sl.length;i++) for(let j=i+1;j<sl.length;j++){
      const a=nodes[sl[i]],b=nodes[sl[j]];
      let dx=b.x-a.x,dy=b.y-a.y,dist=Math.sqrt(dx*dx+dy*dy)||1;
      let f=K_REPEL/(dist*dist),fx=(dx/dist)*f,fy=(dy/dist)*f;
      a.vx-=fx;a.vy-=fy;b.vx+=fx;b.vy+=fy;
    }
    edges.forEach(e=>{const a=nodes[e.src],b=nodes[e.dst];let dx=b.x-a.x,dy=b.y-a.y,dist=Math.sqrt(dx*dx+dy*dy)||1;
      let f=(dist-120)*K_SPRING,fx=(dx/dist)*f,fy=(dy/dist)*f;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;});
    sl.forEach(s=>{nodes[s].vx+=(W/2-nodes[s].x)*0.001;nodes[s].vy+=(H/2-nodes[s].y)*0.001;
      nodes[s].vx*=DAMP;nodes[s].vy*=DAMP;nodes[s].x+=nodes[s].vx;nodes[s].y+=nodes[s].vy;
      nodes[s].x=Math.max(30,Math.min(W-30,nodes[s].x));nodes[s].y=Math.max(30,Math.min(H-30,nodes[s].y));});
  }
  function drawG() {
    ctx.fillStyle='#111';ctx.fillRect(0,0,W,H);
    edges.forEach(e=>{const a=nodes[e.src],b=nodes[e.dst];
      ctx.strokeStyle=e.isPolicy?'#ffd700':'#555';ctx.lineWidth=e.isPolicy?2.5:Math.max(0.5,e.prob*3);
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      const angle=Math.atan2(b.y-a.y,b.x-a.x);const ax=b.x-22*Math.cos(angle),ay=b.y-22*Math.sin(angle);
      ctx.fillStyle=e.isPolicy?'#ffd700':'#888';ctx.beginPath();
      ctx.moveTo(ax+8*Math.cos(angle),ay+8*Math.sin(angle));
      ctx.lineTo(ax+6*Math.cos(angle-2.5),ay+6*Math.sin(angle-2.5));
      ctx.lineTo(ax+6*Math.cos(angle+2.5),ay+6*Math.sin(angle+2.5));ctx.fill();
    });
    MDP_DATA.states.forEach(s=>{const n=nodes[s];const v=values[s]||0;
      const hue=valueToHue(v,vMin,vMax);
      ctx.fillStyle='hsl('+hue+',70%,40%)';
      if(s===MDP_DATA.goalState)ctx.fillStyle='hsl(120,80%,40%)';
      if(s===MDP_DATA.trapState)ctx.fillStyle='hsl(0,80%,35%)';
      ctx.beginPath();ctx.arc(n.x,n.y,18,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.stroke();
      ctx.fillStyle='#fff';ctx.font='bold 10px sans-serif';ctx.textAlign='center';
      ctx.fillText(s,n.x,n.y-3);ctx.font='9px sans-serif';ctx.fillText(v.toFixed(1),n.x,n.y+9);
    });
  }
  let af=0;
  function loop(){simulate();drawG();af++;if(af<300)requestAnimationFrame(loop);}
  loop();
}

// ---- Initialize ----
buildTabs();
</script>
</body>
</html>
)js";

    file.close();
    std::cout << "HTML visualization written to: " << output_path << "\n";
    return true;
}
