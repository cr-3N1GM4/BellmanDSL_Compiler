#!/usr/bin/env python3
"""optimizer.py — Real portfolio rebalancer for MDP-DSL v3.0.

Workflow:
  1. User supplies holdings: [(ticker, qty, avg_buy_price), ...]
  2. We fetch live prices via price_fetcher.py.
  3. We run the compiler on examples/portfolio.mdp to get the regime-aware
     recommended action (one of AggressiveEquity / ModerateEquity / Defensive
     / CashAndGold).
  4. We compute target weights within the recommended action's asset-class
     split, weighted by per-stock Sharpe heuristics.
  5. We emit concrete BUY / SELL / HOLD trades and overall P&L.

Honest scope note: VERIFY guarantees apply to the BUCKET choice (the action
label). The per-stock weighting within the bucket is a heuristic layered on
top of the verified decision.
"""
import json, subprocess, sys
from pathlib import Path

from price_fetcher import PriceFetcher

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = json.loads((ROOT / "portfolio" / "nifty_universe.json").read_text())

# Mapping: recommended-action -> (equity, debt, gold) target shares of NAV.
ACTION_SPLITS = {
    "AggressiveEquity": {"equity": 0.90, "debt": 0.10, "gold": 0.00},
    "ModerateEquity":   {"equity": 0.60, "debt": 0.30, "gold": 0.10},
    "Defensive":        {"equity": 0.30, "debt": 0.40, "gold": 0.30},
    "CashAndGold":      {"equity": 0.10, "debt": 0.40, "gold": 0.50},
}

ASSET_CLASS_BY_SECTOR = {
    "Gold ETF":  "gold",
    "Debt ETF":  "debt",
}

def asset_class(ticker_meta):
    """Bucket a ticker into equity / debt / gold."""
    sec = (ticker_meta or {}).get("sector", "")
    return ASSET_CLASS_BY_SECTOR.get(sec, "equity")


def run_compiler(compiler_path, mdp_file):
    """Run the C++ compiler with --json and return parsed output."""
    proc = subprocess.run(
        [str(compiler_path), str(mdp_file), "--json"],
        capture_output=True, text=True, timeout=20,
    )
    if proc.returncode not in (0, 1):  # 1 is OK (VERIFY may fail)
        raise RuntimeError(f"Compiler failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def find_compiler():
    """Locate the compiler binary in the standard locations."""
    candidates = [ROOT / "bin" / "mdp_compiler",
                  ROOT / "bin" / "mdp_compiler.exe",
                  ROOT / "src" / "mdp_compiler",
                  ROOT / "src" / "mdp_compiler.exe"]
    return next((p for p in candidates if p.exists()), None)


def compute_current_state(holdings, prices, fetcher):
    """Annotate each holding with live price, current value, unrealized P&L."""
    rows = []
    for h in holdings:
        ticker = h["ticker"].upper()
        qty    = float(h.get("qty", 0))
        avg    = float(h.get("avg_buy_price", 0))
        meta   = fetcher.get_stock_meta(ticker) or {}
        price_info = prices.get(ticker, {})
        live = price_info.get("price")
        if live is None:
            rows.append({**h, "ticker": ticker, "live_price": None,
                         "value": 0, "cost": qty * avg,
                         "pnl": 0, "pnl_pct": 0,
                         "asset_class": asset_class(meta), "sector": meta.get("sector", "Unknown"),
                         "beta": meta.get("beta", None), "name": meta.get("name", ticker),
                         "price_source": price_info.get("source", "missing")})
            continue
        value = qty * live
        cost  = qty * avg
        pnl   = value - cost
        rows.append({
            "ticker":      ticker,
            "name":        meta.get("name", ticker),
            "sector":      meta.get("sector", "Unknown"),
            "beta":        meta.get("beta", None),
            "asset_class": asset_class(meta),
            "qty":         qty,
            "avg_buy_price": avg,
            "live_price":  live,
            "cost":        cost,
            "value":       value,
            "pnl":         pnl,
            "pnl_pct":     (pnl / cost * 100.0) if cost > 0 else 0.0,
            "price_source": price_info.get("source", "missing"),
        })
    return rows


def compute_target_weights(action, holdings_rows, fetcher):
    """Translate a recommended action into per-ticker target portfolio weights.

    Strategy:
      - Bucket holdings into asset classes (equity / debt / gold).
      - Allocate target NAV share by ACTION_SPLITS[action].
      - Within equity slice: weight each equity holding by an inverse-beta
        proxy for Sharpe-style risk-adjusted return. (We don't have realised
        Sharpe live; lower beta -> proportionally higher weight in a defensive
        posture, higher beta in an aggressive one.)
      - Within debt slice: distribute evenly across the user's debt ETFs;
        if none, suggest adding LIQUIDBEES.
      - Within gold slice: route to user's GOLDBEES holding; if none and
        gold weight > 0, suggest adding GOLDBEES.

    Returns: dict {ticker: target_weight} summing to ~1.0, plus a list of
             tickers we are RECOMMENDING TO ADD (not currently held).
    """
    split = ACTION_SPLITS.get(action, ACTION_SPLITS["ModerateEquity"])

    by_class = {"equity": [], "debt": [], "gold": []}
    for r in holdings_rows:
        by_class[r["asset_class"]].append(r)

    target = {}
    suggested_additions = []

    # ---- Equity slice: beta-tilted across user equity holdings ----
    eq = by_class["equity"]
    if eq:
        # Risk-aligned tilt: in aggressive mode upweight higher-beta names,
        # in defensive mode upweight lower-beta names. We use a smooth tilt
        # rather than a hard cutoff to keep all holdings represented.
        if action == "AggressiveEquity":
            scores = [max(0.3, (r["beta"] or 1.0))     for r in eq]   # favour higher beta
        elif action == "Defensive":
            scores = [max(0.3, 1.5 - (r["beta"] or 1.0)) for r in eq] # favour lower beta
        else:
            scores = [1.0 for _ in eq]
        total = sum(scores)
        for r, s in zip(eq, scores):
            target[r["ticker"]] = split["equity"] * (s / total)
    elif split["equity"] > 0:
        # User has no equity at all -- suggest NIFTYBEES.
        target["NIFTYBEES"] = split["equity"]
        suggested_additions.append("NIFTYBEES")

    # ---- Debt slice ----
    debt = by_class["debt"]
    if debt:
        for r in debt:
            target[r["ticker"]] = split["debt"] / len(debt)
    elif split["debt"] > 0:
        target["LIQUIDBEES"] = split["debt"]
        suggested_additions.append("LIQUIDBEES")

    # ---- Gold slice ----
    gold = by_class["gold"]
    if gold:
        for r in gold:
            target[r["ticker"]] = split["gold"] / len(gold)
    elif split["gold"] > 0:
        target["GOLDBEES"] = split["gold"]
        suggested_additions.append("GOLDBEES")

    # Normalise (small drift from rounding above).
    total = sum(target.values())
    if total > 0:
        for k in list(target):
            target[k] /= total
    return target, suggested_additions


def compute_trades(holdings_rows, target_weights, fetcher, prices):
    """Compare current vs target allocation -> concrete trades."""
    nav = sum(r["value"] for r in holdings_rows if r["live_price"] is not None)
    if nav <= 0:
        return [], 0.0, []

    # Build a lookup of current quantity per ticker.
    current_qty   = {r["ticker"]: r["qty"]        for r in holdings_rows}
    current_value = {r["ticker"]: r["value"]      for r in holdings_rows}
    live_price    = {r["ticker"]: r["live_price"] for r in holdings_rows
                                                   if r["live_price"] is not None}

    # For tickers in target_weights that the user doesn't yet hold,
    # we need a live price -- the fetcher gives us seed prices as fallback.
    extra = [t for t in target_weights if t not in current_qty]
    if extra:
        extra_prices = fetcher.fetch(extra)
        for t, info in extra_prices.items():
            if info.get("price"):
                live_price[t] = info["price"]
                prices.setdefault(t, info)

    trades = []
    for ticker, tw in target_weights.items():
        target_value = tw * nav
        current_v    = current_value.get(ticker, 0)
        diff_value   = target_value - current_v
        price        = live_price.get(ticker)
        if price is None or price <= 0:
            continue
        diff_qty     = diff_value / price
        action_label = "BUY" if diff_qty > 0 else ("SELL" if diff_qty < 0 else "HOLD")
        # Round qty to nearest integer share. Skip trades of < 0.5 shares to
        # cut noise.
        diff_qty_round = round(diff_qty)
        if diff_qty_round == 0:
            action_label = "HOLD"
        if abs(diff_qty) < 0.5 and current_v > 0:
            continue
        meta = fetcher.get_stock_meta(ticker) or {}
        trades.append({
            "ticker":       ticker,
            "name":         meta.get("name", ticker),
            "sector":       meta.get("sector", "Unknown"),
            "action":       action_label,
            "current_qty":  current_qty.get(ticker, 0),
            "target_qty":   round((target_value / price)) if price > 0 else 0,
            "delta_qty":    diff_qty_round,
            "delta_value":  round(diff_value, 2),
            "live_price":   price,
            "target_weight":  round(tw,    4),
            "current_weight": round(current_v / nav, 4) if nav > 0 else 0,
        })

    # Sort: biggest abs(delta_value) first so user sees most impactful trades up top.
    trades.sort(key=lambda t: -abs(t["delta_value"]))
    return trades, nav, []


def optimize(holdings, compiler_path=None, mdp_file=None, fetcher=None):
    """End-to-end: holdings in, full plan out.

    holdings: list of dicts: {"ticker": "TCS", "qty": 30, "avg_buy_price": 3500}
    """
    fetcher       = fetcher or PriceFetcher()
    compiler_path = compiler_path or find_compiler()
    mdp_file      = mdp_file      or (ROOT / "examples" / "portfolio.mdp")

    if compiler_path is None or not Path(compiler_path).exists():
        raise RuntimeError(
            "Compiler binary not found. Run ./build.sh (Linux/macOS) or BUILD.bat (Windows) first."
        )
    if not Path(mdp_file).exists():
        raise RuntimeError(f"MDP file not found: {mdp_file}")

    # Step 1: prices for the user's tickers.
    tickers = [h["ticker"].upper() for h in holdings]
    prices  = fetcher.fetch(tickers)

    # Step 2: current portfolio state.
    rows = compute_current_state(holdings, prices, fetcher)

    # Step 3: run the compiler.
    compiler_output = run_compiler(compiler_path, mdp_file)
    recommended_action = compiler_output.get("policy", {}).get(
        # Use the most-likely regime from the inferred HMM means: most-likely
        # is whatever the HMM transition-matrix stationary distribution favours;
        # for now we just take the regime with the highest pi(s) in compiler_output.
        # Heuristic: use the regime with the highest emission-mean if the
        # compiler_output exposes it, otherwise the highest V*.
        max(compiler_output["values"], key=compiler_output["values"].get),
        "ModerateEquity",
    )

    # Step 4: target weights.
    target_weights, additions = compute_target_weights(
        recommended_action, rows, fetcher)

    # Step 5: trades.
    trades, nav, _ = compute_trades(rows, target_weights, fetcher, prices)

    # Roll up portfolio-level stats.
    total_cost  = sum(r["cost"]  for r in rows)
    total_value = sum(r["value"] for r in rows if r["live_price"] is not None)
    total_pnl   = total_value - total_cost

    return {
        "holdings":     rows,
        "trades":       trades,
        "nav":          round(nav, 2),
        "total_cost":   round(total_cost, 2),
        "total_value":  round(total_value, 2),
        "total_pnl":    round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / total_cost * 100.0), 2) if total_cost > 0 else 0,
        "recommended_action": recommended_action,
        "target_weights":     target_weights,
        "suggested_additions": additions,
        "verify_assertions": [
            {"label": "Bull -> AggressiveEquity",
             "expr":  "pi(Bull) == AggressiveEquity",
             "passed": compiler_output["policy"].get("Bull")   == "AggressiveEquity"},
            {"label": "Bear -> Defensive",
             "expr":  "pi(Bear) == Defensive",
             "passed": compiler_output["policy"].get("Bear")   == "Defensive"},
            {"label": "Crisis -> CashAndGold",
             "expr":  "pi(Crisis) == CashAndGold",
             "passed": compiler_output["policy"].get("Crisis") == "CashAndGold"},
            {"label": "V(Bull) > V(Bear)",
             "expr":  "V(Bull) > V(Bear)",
             "passed": compiler_output["values"]["Bull"]  >  compiler_output["values"]["Bear"]},
            {"label": "V(Crisis) < V(Sideways)",
             "expr":  "V(Crisis) < V(Sideways)",
             "passed": compiler_output["values"]["Crisis"] < compiler_output["values"]["Sideways"]},
        ],
        "hmm":          compiler_output.get("hmm"),
        "policy":       compiler_output.get("policy"),
        "values":       compiler_output.get("values"),
        "compiler_version": compiler_output.get("version", "3.0"),
    }


if __name__ == "__main__":
    # Quick CLI demo.
    sample = [
        {"ticker": "RELIANCE", "qty": 50, "avg_buy_price": 2400},
        {"ticker": "TCS",      "qty": 30, "avg_buy_price": 3500},
        {"ticker": "HDFCBANK", "qty": 40, "avg_buy_price": 1600},
        {"ticker": "INFY",     "qty": 25, "avg_buy_price": 1800},
        {"ticker": "GOLDBEES", "qty": 200,"avg_buy_price":   65},
    ]
    print("Sample holdings:")
    for h in sample: print(f"  {h['ticker']:12s} {h['qty']:>4} @ Rs {h['avg_buy_price']}")
    print()
    result = optimize(sample)
    print(f"Current NAV:    Rs {result['nav']:,.2f}")
    print(f"Total cost:     Rs {result['total_cost']:,.2f}")
    print(f"Total P&L:      Rs {result['total_pnl']:+,.2f}  ({result['total_pnl_pct']:+.2f}%)")
    print(f"\nRecommended action: {result['recommended_action']}")
    print(f"\nProposed trades (top by impact):")
    for t in result["trades"]:
        sign = "+" if t["delta_qty"] > 0 else ""
        print(f"  {t['action']:5s}  {t['ticker']:11s}  "
              f"qty {sign}{t['delta_qty']:>4}  (target {t['target_qty']:>4})  "
              f"Rs {t['delta_value']:>+12,.2f}")
    print(f"\nVERIFY proofs: "
          f"{sum(1 for a in result['verify_assertions'] if a['passed'])} / "
          f"{len(result['verify_assertions'])} passing")
