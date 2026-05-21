#!/usr/bin/env python3
"""build_demo_data.py — Generate demo_data.json from the compiler.

This is the standalone data-builder that the portfolio dashboard reads in
demo mode. It runs the C++ compiler on portfolio.mdp, parses the JSON
output, augments it with Indian-market context (annotations, sector colors,
synthetic regime history), and writes demo_data.json.

Run from the repo root:   python3 portfolio/build_demo_data.py

The dashboard works WITHOUT this script because we ship a pre-built
demo_data.json in the bundle. Re-run this script if you want fresh numbers.
"""
import json, math, os, subprocess, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPILER = ROOT / "bin" / "mdp_compiler"
if not COMPILER.exists():
    COMPILER = ROOT / "src" / "mdp_compiler"           # fallback for raw build
if not COMPILER.exists():
    print("[ERROR] Compiler binary not found at bin/mdp_compiler or src/mdp_compiler.")
    print("        Run ./build.sh first.")
    sys.exit(1)

MDP_FILE = ROOT / "examples" / "portfolio.mdp"
OUT_FILE = ROOT / "portfolio" / "demo_data.json"

print(f"Running compiler: {COMPILER} {MDP_FILE}")
proc = subprocess.run(
    [str(COMPILER), str(MDP_FILE), "--json"],
    capture_output=True, text=True, cwd=str(ROOT)
)
if proc.returncode != 0:
    # Verify failures are exit code 1; we still want the JSON though.
    print(f"[WARN] Compiler exited with code {proc.returncode}.")
    print(proc.stderr[-2000:])
try:
    comp = json.loads(proc.stdout)
except json.JSONDecodeError as e:
    print(f"[ERROR] Compiler JSON parse failed: {e}")
    print(proc.stdout[:1000])
    sys.exit(1)

# ---- Augment with Indian-market context ----
# Sample NSE stock universe (sector-colored).
stocks = [
    {"ticker": "TCS",       "sector": "IT",       "beta": 0.85, "sharpe_bull": 1.2, "color": "#56cbf9"},
    {"ticker": "INFY",      "sector": "IT",       "beta": 0.92, "sharpe_bull": 1.1, "color": "#56cbf9"},
    {"ticker": "HDFCBANK",  "sector": "Banking",  "beta": 1.05, "sharpe_bull": 1.3, "color": "#10b981"},
    {"ticker": "RELIANCE",  "sector": "Energy",   "beta": 0.98, "sharpe_bull": 0.9, "color": "#f2c94c"},
    {"ticker": "GOLDBEES",  "sector": "Gold",     "beta":-0.10, "sharpe_bull":-0.2, "color": "#ffd700"},
]

# Synthetic 5-year regime history with annotation hooks.
# (60 monthly samples; in production this would come from Viterbi decoding.)
random.seed(42)
regime_seq = []
state = "Sideways"
A_named = {
    "Bull":     {"Bull":0.70, "Bear":0.10, "Sideways":0.15, "Crisis":0.05},
    "Bear":     {"Bull":0.05, "Bear":0.70, "Sideways":0.20, "Crisis":0.05},
    "Sideways": {"Bull":0.10, "Bear":0.10, "Sideways":0.78, "Crisis":0.02},
    "Crisis":   {"Bull":0.05, "Bear":0.25, "Sideways":0.05, "Crisis":0.65},
}
months = ["2020-01","2020-02","2020-03","2020-04","2020-05","2020-06","2020-07","2020-08","2020-09","2020-10","2020-11","2020-12",
          "2021-01","2021-02","2021-03","2021-04","2021-05","2021-06","2021-07","2021-08","2021-09","2021-10","2021-11","2021-12",
          "2022-01","2022-02","2022-03","2022-04","2022-05","2022-06","2022-07","2022-08","2022-09","2022-10","2022-11","2022-12",
          "2023-01","2023-02","2023-03","2023-04","2023-05","2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12",
          "2024-01","2024-02","2024-03","2024-04","2024-05","2024-06","2024-07","2024-08","2024-09","2024-10","2024-11","2024-12"]
# Force some known regimes at iconic months.
force = {
    "2020-03": "Crisis",   # COVID
    "2020-04": "Crisis",
    "2020-05": "Bear",
    "2020-11": "Bull",     # vaccine rally
    "2022-01": "Bear",     # FII selloff
    "2022-02": "Bear",
}
nifty = 12000.0
prices = []
for m in months:
    if m in force: state = force[m]
    else:
        # weighted regime transition
        r = random.random(); acc = 0.0
        for n, p in A_named[state].items():
            acc += p
            if r < acc: state = n; break
    # Synthetic monthly return per regime.
    mu = {"Bull":0.025, "Bear":-0.025, "Sideways":0.005, "Crisis":-0.08}[state]
    sd = {"Bull":0.025, "Bear":0.03,   "Sideways":0.02,  "Crisis":0.06}[state]
    r = random.gauss(mu, sd)
    nifty *= (1 + r)
    regime_seq.append({"month": m, "regime": state, "nifty": round(nifty, 1), "return": round(r*100, 2)})

annotations = [
    {"month": "2020-03", "label": "COVID Crash",        "type": "Crisis"},
    {"month": "2020-11", "label": "Vaccine Rally",      "type": "Bull"},
    {"month": "2022-01", "label": "FII Selloff Begins", "type": "Bear"},
]

# Compute concrete stock weights for the optimal action (ModerateEquity here,
# given the current belief leans Bull/Sideways).
# Recipe per master spec: 60% equity / 30% debt / 10% gold for ModerateEquity.
# Within equity, weight by stock Sharpe (truncated, normalised).
def stock_weights(action_label):
    rules = {
        "AggressiveEquity":  (0.90, 0.10, 0.00),   # equity / debt / gold
        "ModerateEquity":    (0.60, 0.30, 0.10),
        "Defensive":         (0.30, 0.40, 0.30),
        "CashAndGold":       (0.10, 0.40, 0.50),
    }
    eq, debt, gold = rules.get(action_label, (0.60, 0.30, 0.10))
    eq_stocks = [s for s in stocks if s["sector"] != "Gold"]
    total_sharpe = sum(max(0.1, s["sharpe_bull"]) for s in eq_stocks)
    weights = {}
    for s in eq_stocks:
        w = (s["sharpe_bull"] / total_sharpe) * eq
        weights[s["ticker"]] = round(max(0.0, w), 4)
    weights["GOLDBEES"] = round(gold, 4)
    weights["Debt"]     = round(debt, 4)
    # Normalise rounding error.
    total = sum(weights.values())
    if total > 0:
        for k in weights: weights[k] = round(weights[k] / total, 4)
    return weights

# Compute current belief by running a Viterbi-like rolling estimate.
# For demo purposes we just use the last 6 months' regime frequencies as
# a stand-in for the posterior belief.
recent = [r["regime"] for r in regime_seq[-12:]]
belief = {r: recent.count(r) / len(recent) for r in ["Bull","Bear","Sideways","Crisis"]}
for r in ["Bull","Bear","Sideways","Crisis"]:
    belief.setdefault(r, 0.0)
# Map the belief to a recommended action via simple argmax over regime-action Q.
# Just use the compiler's pi(s) for the most-likely current regime.
most_likely = max(belief.items(), key=lambda x: x[1])[0]
recommended_action = comp["policy"].get(most_likely, "ModerateEquity")
weights = stock_weights(recommended_action)

demo = {
    "hmm": comp["hmm"],
    "values": comp["values"],
    "policy": comp["policy"],
    "verify_failures": comp["verify_failures"],
    "stocks": stocks,
    "regime_history": regime_seq,
    "annotations": annotations,
    "current_belief": belief,
    "most_likely_regime": most_likely,
    "recommended_action": recommended_action,
    "weights": weights,
    "verify_assertions": [
        {"label": "Bull regime → AggressiveEquity",
         "expr":  "π(Bull) == AggressiveEquity",
         "passed": comp["policy"].get("Bull")     == "AggressiveEquity"},
        {"label": "Bear regime → Defensive allocation",
         "expr":  "π(Bear) == Defensive",
         "passed": comp["policy"].get("Bear")     == "Defensive"},
        {"label": "Crisis regime → CashAndGold",
         "expr":  "π(Crisis) == CashAndGold",
         "passed": comp["policy"].get("Crisis")   == "CashAndGold"},
        {"label": "V(Bull) > V(Bear)",
         "expr":  "V(Bull) > V(Bear)",
         "passed": comp["values"]["Bull"]  >  comp["values"]["Bear"]},
        {"label": "V(Crisis) < V(Sideways)",
         "expr":  "V(Crisis) < V(Sideways)",
         "passed": comp["values"]["Crisis"] < comp["values"]["Sideways"]},
    ],
    "discount": comp.get("discount", 0.92),
    "solver_mode": comp.get("solver_mode", "policy_iteration"),
    "portfolio_size_inr": 1000000,
    "generated_from": "examples/portfolio.mdp via mdp_compiler v3.0",
}

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(demo, f, indent=2)
print(f"Wrote {OUT_FILE}")

# Also inject into the dashboard HTML if the placeholder is still present.
# This makes the dashboard self-contained (demo mode works offline).
DASHBOARD = ROOT / "portfolio" / "portfolio_dashboard.html"
if DASHBOARD.exists():
    html = DASHBOARD.read_text(encoding="utf-8")
    placeholder = "/*DEMO_JSON_INJECTION_POINT*/null"
    if placeholder in html:
        # Fresh placeholder still in file — first build.
        html = html.replace(placeholder, json.dumps(demo))
        print(f"Injected demo data into {DASHBOARD}")
    else:
        # Already injected — replace the entire DEMO_DATA = {...} assignment.
        import re
        new_html, n = re.subn(
            r"const DEMO_DATA = .*?;",
            "const DEMO_DATA = " + json.dumps(demo).replace("\\", "\\\\") + ";",
            html, count=1, flags=re.DOTALL
        )
        if n == 1:
            html = new_html
            print(f"Refreshed embedded DEMO_DATA in {DASHBOARD}")
        else:
            print(f"[WARN] Could not locate DEMO_DATA in {DASHBOARD}; leaving HTML untouched.")
    DASHBOARD.write_text(html, encoding="utf-8")

print(f"  HMM regimes: {len(comp['hmm']['states'])}")
print(f"  VERIFY pass: {sum(1 for a in demo['verify_assertions'] if a['passed'])}"
      f" / {len(demo['verify_assertions'])}")
print(f"  Recommended: {recommended_action}")
