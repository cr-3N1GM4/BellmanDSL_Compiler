#!/usr/bin/env bash
# build.sh — one-command build for MDP-DSL v3.0 Phase 3B.
# Produces:
#   bin/mdp_compiler   — main compiler with all phase flags
#                          (--animate, --html, --json, --diagnose,
#                           --fragility, --repair, --show-hmm)
#   bin/mdp_autopsy    — failure-log backwards solver (Phase 3B)
set -e

cd "$(dirname "$0")"

echo "=== MDP-DSL v3.0 Phase 3B build ==="
echo "Working directory: $(pwd)"
echo

# ---- preflight ----
for f in src/mdp_compiler.cpp src/mdp_autopsy.cpp \
         src/visualizer.hpp src/visualizer.cpp \
         src/html_output.hpp src/html_output.cpp; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing $f"; exit 1
    fi
done

if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found on PATH."
    echo "        Install build-essential (Debian/Ubuntu) or Xcode CLI tools (macOS)."
    exit 1
fi

mkdir -p bin

# ---- build mdp_compiler ----
cd src
g++ -std=c++17 -O2 -DMDP_VIZ_ENABLED -o ../bin/mdp_compiler mdp_compiler.cpp
cd ..
echo "[OK] Built: bin/mdp_compiler"

# ---- build mdp_autopsy (Phase 3B) ----
cd src
g++ -std=c++17 -O2 -o ../bin/mdp_autopsy mdp_autopsy.cpp
cd ..
echo "[OK] Built: bin/mdp_autopsy"
echo

# ---- smoke tests ----
echo "Smoke tests (VERIFY blocks):"
./bin/mdp_compiler examples/tiger.mdp     | grep -E 'VERIFY (PASS|FAIL)' | sed 's/^/  /'
echo
./bin/mdp_compiler examples/robot_nav.mdp | grep -E 'VERIFY (PASS|FAIL)' | sed 's/^/  /'
echo
./bin/mdp_compiler examples/portfolio.mdp | grep -E 'VERIFY (PASS|FAIL)' | sed 's/^/  /'

echo
echo "Smoke tests (Phase 3A autopsy --diagnose):"
./bin/mdp_compiler examples/autopsy_demo/myopic_corridor.mdp     --diagnose 2>&1 \
    | grep -E '\[(ERROR|WARN)/' | head -2 | sed 's/^/  /'
./bin/mdp_compiler examples/autopsy_demo/dead_state_demo.mdp     --diagnose=structural 2>&1 \
    | grep -E '\[(ERROR|WARN)/' | head -1 | sed 's/^/  /'
./bin/mdp_compiler examples/autopsy_demo/reward_hacking_demo.mdp --diagnose=reward 2>&1 \
    | grep -E '\[(ERROR|WARN)/' | head -1 | sed 's/^/  /'
./bin/mdp_compiler examples/autopsy_demo/broken_for_repair.mdp   --repair "pi(Start) == Move" 2>&1 \
    | grep -E 'Minimal fix' | sed 's/^/  /'

echo
echo "Smoke tests (Phase 3B mdp_autopsy):"
./bin/mdp_autopsy examples/autopsy_demo/medical_dose.mdp examples/autopsy_demo/medical_clean.log 2>&1 \
    | grep 'MODEL_' | head -1 | sed 's/^/  medical_clean.log    -> /'
./bin/mdp_autopsy examples/autopsy_demo/medical_dose.mdp examples/autopsy_demo/medical_run.log 2>&1 \
    | grep 'MODEL_' | head -1 | sed 's/^/  medical_run.log      -> /'
./bin/mdp_autopsy examples/autopsy_demo/robot_model.mdp examples/autopsy_demo/robot_run.log 2>&1 \
    | grep 'MODEL_' | head -1 | sed 's/^/  robot_run.log        -> /'

echo
echo "Try next:"
echo "  ./bin/mdp_compiler examples/portfolio.mdp --show-hmm"
echo "  ./bin/mdp_compiler examples/autopsy_demo/broken_for_repair.mdp --repair \"pi(Start) == Move\""
echo "  ./bin/mdp_autopsy   examples/autopsy_demo/robot_model.mdp examples/autopsy_demo/robot_run.log"
echo "  ./bin/mdp_autopsy   examples/portfolio.mdp portfolio/portfolio_run.log --json"
echo "  python3 portfolio/portfolio_server.py             # then visit http://localhost:8421"
echo "  # In the dashboard: click 'Run Autopsy' under the VERIFY block"
