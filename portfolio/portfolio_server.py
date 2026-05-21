#!/usr/bin/env python3
"""portfolio_server.py — Minimal stdlib HTTP server for the MDP-DSL dashboard.

GET  /                  → serves the dashboard HTML
GET  /demo              → serves the pre-built demo_data.json
GET  /universe          → serves nifty_universe.json (ticker autocomplete)
GET  /portfolio_dashboard.html → same as /
POST /optimize          → runs the compiler on portfolio.mdp, returns JSON
POST /portfolio/optimize → real-portfolio rebalancer (Phase 2.5)

Run from the repo root:
    python3 portfolio/portfolio_server.py

Then open  http://localhost:8421  in a browser.

This file uses ONLY the Python standard library — no Flask, no FastAPI.
Per the master spec: 'minimal stdlib HTTP server — no Flask required'.
"""
import json, os, subprocess, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# Add the portfolio directory to sys.path so optimizer + price_fetcher import cleanly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PORT = 8421
ROOT = HERE.parent
DASHBOARD_HTML = ROOT / "portfolio" / "portfolio_dashboard.html"
DEMO_JSON      = ROOT / "portfolio" / "demo_data.json"
UNIVERSE_JSON  = ROOT / "portfolio" / "nifty_universe.json"
MDP_FILE       = ROOT / "examples" / "portfolio.mdp"
COMPILER_PATHS = [ROOT / "bin" / "mdp_compiler",
                  ROOT / "bin" / "mdp_compiler.exe",
                  ROOT / "src" / "mdp_compiler",
                  ROOT / "src" / "mdp_compiler.exe"]
COMPILER = next((p for p in COMPILER_PATHS if p.exists()), None)

# Lazy-load the optimizer so a missing yfinance install doesn't kill the server.
_OPTIMIZER = None
def get_optimizer():
    global _OPTIMIZER
    if _OPTIMIZER is None:
        import optimizer
        _OPTIMIZER = optimizer
    return _OPTIMIZER

# Simple 15-minute cache for compiler runs (per the spec's Yahoo Finance trap).
_CACHE = {"ts": 0, "value": None}
CACHE_TTL = 15 * 60


def run_compiler():
    """Run the compiler on examples/portfolio.mdp and return parsed JSON."""
    if COMPILER is None:
        return None
    try:
        proc = subprocess.run(
            [str(COMPILER), str(MDP_FILE), "--json"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=20,
        )
        return json.loads(proc.stdout)
    except Exception as e:
        sys.stderr.write(f"[server] compiler run failed: {e}\n")
        return None


class Handler(BaseHTTPRequestHandler):
    # Quiet down the default access-log spam.
    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Not found: {path.name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        # CORS preflight.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/portfolio_dashboard.html"):
            self._send_file(DASHBOARD_HTML, "text/html; charset=utf-8")
            return
        if path == "/demo":
            self._send_file(DEMO_JSON, "application/json; charset=utf-8")
            return
        if path == "/universe":
            self._send_file(UNIVERSE_JSON, "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send_json({"ok": True, "compiler": str(COMPILER) if COMPILER else None})
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        # ---- /portfolio/autopsy: Phase 3B failure-log autopsy ----
        if path == "/portfolio/autopsy":
            # Regenerate the portfolio.log from the latest demo data, then
            # invoke mdp_autopsy with --json and return its parsed verdict.
            log_path = ROOT / "portfolio" / "portfolio_run.log"
            try:
                # Build the log fresh each run so it reflects current demo data.
                builder = ROOT / "portfolio" / "build_portfolio_log.py"
                if builder.exists():
                    subprocess.run([sys.executable, str(builder)],
                                   check=True, capture_output=True, cwd=str(ROOT), timeout=10)
                if not log_path.exists():
                    self._send_json({"error": f"Log not generated: {log_path}"}, 500)
                    return
                # Locate the mdp_autopsy binary.
                autopsy_bin = next((p for p in [
                    ROOT / "bin" / "mdp_autopsy",
                    ROOT / "bin" / "mdp_autopsy.exe",
                ] if p.exists()), None)
                if autopsy_bin is None:
                    self._send_json({"error": "mdp_autopsy binary not found. Build it first."}, 500)
                    return
                proc = subprocess.run(
                    [str(autopsy_bin), str(MDP_FILE), str(log_path), "--json"],
                    capture_output=True, text=True, cwd=str(ROOT), timeout=30,
                )
                # mdp_autopsy returns 0/1/2 for verdicts; 3 for parse error.
                if proc.returncode == 3:
                    self._send_json({"error": "Autopsy parse error: " + proc.stderr.strip()}, 500)
                    return
                result = json.loads(proc.stdout)
                self._send_json(result)
            except subprocess.TimeoutExpired:
                self._send_json({"error": "Autopsy timed out (30s)."}, 500)
            except Exception as e:
                sys.stderr.write(f"[server] /portfolio/autopsy failed: {e}\n")
                self._send_json({"error": str(e)}, 500)
            return

        # ---- /portfolio/optimize: REAL portfolio rebalancer ----
        if path == "/portfolio/optimize":
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._send_json({"error": "Invalid JSON body"}, 400)
                return
            holdings = req.get("holdings", [])
            if not isinstance(holdings, list) or len(holdings) == 0:
                self._send_json({"error": "holdings is required (list)"}, 400)
                return
            try:
                opt = get_optimizer()
                result = opt.optimize(holdings, compiler_path=COMPILER, mdp_file=MDP_FILE)
                self._send_json(result)
            except Exception as e:
                sys.stderr.write(f"[server] /portfolio/optimize failed: {e}\n")
                self._send_json({"error": str(e)}, 500)
            return

        # ---- /optimize: strategy-only demo refresh ----
        if path == "/optimize":
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length: self.rfile.read(length)

            now = time.time()
            if _CACHE["value"] and (now - _CACHE["ts"]) < CACHE_TTL:
                self._send_json(_CACHE["value"])
                return

            comp = run_compiler()
            if comp is None:
                try:
                    comp = json.loads(DEMO_JSON.read_text())
                except Exception:
                    self._send_json({"error": "Compiler unavailable and no demo data."}, 500)
                    return

            _CACHE.update({"ts": now, "value": comp})
            self._send_json(comp)
            return

        self.send_error(404)


def main():
    if not DASHBOARD_HTML.exists():
        sys.stderr.write(f"[ERROR] Dashboard not found: {DASHBOARD_HTML}\n"
                         f"        Did you unpack the bundle in the right place?\n")
        sys.exit(1)
    if COMPILER:
        print(f"[server] Using compiler: {COMPILER}")
    else:
        print(f"[server] Compiler not found in {[str(p) for p in COMPILER_PATHS]}.")
        print(f"[server] Will serve demo data only.")

    httpd = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  MDP-DSL Portfolio Server")
    print(f"  ────────────────────────────────────")
    print(f"  Dashboard:        {url}")
    print(f"  Demo data:        {url}/demo")
    print(f"  Universe:         {url}/universe")
    print(f"  Strategy refresh: POST {url}/optimize")
    print(f"  Real rebalancer:  POST {url}/portfolio/optimize")
    print(f"  Autopsy:          POST {url}/portfolio/autopsy")
    print(f"\n  Press Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")


if __name__ == "__main__":
    main()
