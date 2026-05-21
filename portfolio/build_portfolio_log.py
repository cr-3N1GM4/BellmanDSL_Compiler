#!/usr/bin/env python3
"""build_portfolio_log.py — Generate a portfolio.log from regime history.

The log encodes the observed sequence of (regime, recommended-action) tuples
plus the realized monthly return. Each step's "reward" is the Sharpe-adjusted
historical return.

This file is consumed by mdp_autopsy to compare the HMM-learned transition
matrix against the empirical regime transitions, producing a verdict on
whether the strategy is MODEL_ACCURATE / OPTIMISTIC / DANGEROUS.

Usage:
    python3 portfolio/build_portfolio_log.py
    # writes portfolio/portfolio_run.log
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "portfolio" / "demo_data.json"
OUT  = ROOT / "portfolio" / "portfolio_run.log"

if not DEMO.exists():
    print(f"[ERROR] {DEMO} not found. Run build_demo_data.py first.")
    sys.exit(1)

d = json.loads(DEMO.read_text())
hist   = d["regime_history"]
policy = d["policy"]

# We emit one log step per month: state = regime, action = policy[regime].
# Reward = the realized monthly return (already in % units).
lines = ["LOG:"]
for i, row in enumerate(hist, 1):
    regime = row["regime"]
    action = policy.get(regime, "ModerateEquity")
    ret    = row["return"]
    lines.append(f"  step {i:>3}  state {regime:<10}  action {action:<18}  reward {ret:.4f}")

# Detect FAILURE: any month with return < -15% while AggressiveEquity was
# recommended is a strategy-level failure. (Per the spec.)
failure_found = False
for i, row in enumerate(hist):
    regime = row["regime"]
    action = policy.get(regime, "ModerateEquity")
    if row["return"] < -15.0 and action == "AggressiveEquity":
        failure_found = True
        break

# Even if no aggressive-equity drawdown occurred, the synthetic regime
# history is long enough that Crisis -> bad return sequences usually count
# as a soft failure. Append FAILURE iff any month had return below -15%
# (the strategy clearly faced a significant adverse event).
if any(row["return"] < -15.0 for row in hist):
    failure_found = True
    lines.append("  FAILURE")

OUT.write_text("\n".join(lines) + "\n")
print(f"Wrote {OUT}")
print(f"  Months:    {len(hist)}")
print(f"  FAILURE:   {failure_found}")
worst = min(hist, key=lambda r: r["return"])
print(f"  Worst:     {worst['month']}  {worst['regime']}  return {worst['return']:+.2f}%")
