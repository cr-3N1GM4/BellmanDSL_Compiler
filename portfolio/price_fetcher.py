#!/usr/bin/env python3
"""price_fetcher.py — Live NSE price fetcher with caching + offline fallback.

Tries yfinance (if installed and Yahoo is reachable). Falls back to a baked-in
seed price snapshot from nifty_universe.json if the network is down. Caches
fresh fetches to portfolio/.price_cache.json with a 15-minute TTL (per the
master spec's Yahoo-rate-limit trap).

Public API:
    fetcher = PriceFetcher()
    prices  = fetcher.fetch(['RELIANCE', 'TCS', 'GOLDBEES'])
        -> {'RELIANCE': {'price': 1267.45, 'source': 'yfinance', 'fetched_at': 1700000000.0}, ...}
"""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = ROOT / "portfolio" / "nifty_universe.json"
CACHE_FILE    = ROOT / "portfolio" / ".price_cache.json"
CACHE_TTL_S   = 15 * 60   # 15 minutes


class PriceFetcher:
    def __init__(self):
        self.universe = json.loads(UNIVERSE_FILE.read_text())
        self.seed_prices = self.universe["_seed_prices"]
        self._yf_module = None
        self._tried_yf  = False
        self._load_cache()

    # ---- Cache ----
    def _load_cache(self):
        try:
            self.cache = json.loads(CACHE_FILE.read_text())
            if not isinstance(self.cache, dict):
                self.cache = {}
        except Exception:
            self.cache = {}

    def _save_cache(self):
        try:
            CACHE_FILE.write_text(json.dumps(self.cache, indent=2))
        except Exception as e:
            sys.stderr.write(f"[price_fetcher] cache write failed: {e}\n")

    def _cached_fresh(self, ticker):
        entry = self.cache.get(ticker)
        if not entry: return None
        if (time.time() - entry.get("fetched_at", 0)) > CACHE_TTL_S:
            return None
        return entry

    # ---- yfinance lazy import ----
    def _get_yfinance(self):
        if self._tried_yf:
            return self._yf_module
        self._tried_yf = True
        try:
            import yfinance
            self._yf_module = yfinance
        except ImportError:
            sys.stderr.write("[price_fetcher] yfinance not installed; using seed prices.\n")
            sys.stderr.write("[price_fetcher] Install with: pip install yfinance\n")
            self._yf_module = None
        return self._yf_module

    def _yf_ticker(self, ticker):
        """Return the yfinance-formatted symbol for an NSE ticker."""
        for s in self.universe["stocks"]:
            if s["ticker"] == ticker:
                # yf field may need URL-decoding (e.g. M%26M -> M&M).
                return s["yf"].replace("%26", "&")
        # Unknown ticker: try the .NS convention.
        return ticker + ".NS"

    def _fetch_one_live(self, ticker):
        """Try a live fetch via yfinance. Returns price (float) or None."""
        yf = self._get_yfinance()
        if yf is None:
            return None
        # Silence yfinance's noisy logging on transient failures (the
        # spec calls for a Yahoo rate-limit trap — we handle it via the
        # cache + fallback, not stderr spam).
        import logging
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        try:
            symbol = self._yf_ticker(ticker)
            t = yf.Ticker(symbol)
            # fast_info is the cheapest call; fall back to info if needed.
            try:
                p = float(t.fast_info["lastPrice"])
                if p > 0: return p
            except Exception:
                pass
            try:
                info = t.info
                for key in ("currentPrice", "regularMarketPrice", "previousClose"):
                    if info.get(key):
                        return float(info[key])
            except Exception:
                pass
            # Last resort: 1-day history.
            try:
                hist = t.history(period="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
        except Exception:
            pass
        return None

    # ---- Public ----
    def fetch(self, tickers):
        """Fetch live prices for a list of tickers.

        Returns dict mapping ticker -> {price, source, fetched_at, ticker}.
        Source is one of 'yfinance' / 'cache' / 'seed'.
        """
        out = {}
        now = time.time()
        for t in tickers:
            t = t.strip().upper()
            if not t: continue

            cached = self._cached_fresh(t)
            if cached:
                out[t] = {**cached, "ticker": t, "source": "cache"}
                continue

            live = self._fetch_one_live(t)
            if live is not None:
                entry = {"price": live, "fetched_at": now, "source": "yfinance"}
                self.cache[t] = entry
                out[t] = {**entry, "ticker": t}
                continue

            # Fall back to seed price.
            seed = self.seed_prices.get(t)
            if seed is not None:
                out[t] = {"price": float(seed), "fetched_at": 0,
                          "source": "seed", "ticker": t}
            else:
                out[t] = {"price": None, "fetched_at": 0,
                          "source": "unknown", "ticker": t,
                          "error": "Ticker not in universe and no live price."}
        self._save_cache()
        return out

    def get_stock_meta(self, ticker):
        ticker = ticker.upper()
        for s in self.universe["stocks"]:
            if s["ticker"] == ticker:
                return s
        return None


if __name__ == "__main__":
    # CLI sanity-check.  Usage:  python3 price_fetcher.py RELIANCE TCS GOLDBEES
    tickers = sys.argv[1:] or ["RELIANCE", "TCS", "GOLDBEES"]
    f = PriceFetcher()
    print(f"Fetching {tickers}...")
    out = f.fetch(tickers)
    for t in tickers:
        d = out.get(t.upper(), {})
        price = d.get("price")
        if price is None:
            print(f"  {t:12s}  N/A         (source={d.get('source')})")
        else:
            age = (time.time() - d.get("fetched_at", 0))
            age_s = "" if d.get("source") == "seed" else f"  (age {age:.0f}s)"
            print(f"  {t:12s}  Rs {price:>9.2f}  (source={d.get('source')}){age_s}")
