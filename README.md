# MDP-DSL

> A compiled domain-specific language for Markov Decision Processes
> and their partially observable extensions.

MDP-DSL lets you write a decision-theoretic specification (states,
actions, transitions, rewards, observations) as a clean text file
and then have a compiler check it, solve it, and verify formal
properties of the resulting policy. No external dependencies. One
g++ command to build.

This is a small project, offered as a contribution toward
compiler-first tooling for sequential decisions. If you find it
useful or find a bug, I'd love to hear from you.

---

## What it does

* **Compiles** a `.mdp` specification through a five-phase pipeline
  (preprocessor, parser, validator, solver, verifier).
* **Validates** the specification against 25 named semantic checks
  (probability sums, reference resolution, POMDP constraints, etc.)
  before any solver runs.
* **Solves** with one of four backends: Value Iteration, Policy
  Iteration, Q-Learning, or PBVI (Point-Based Value Iteration for
  POMDPs).
* **Bridges HMMs**: Hidden Markov Models can be fitted on a CSV via
  Baum-Welch and their learned transition matrix plumbed into the
  MDP as a first-class language construct.
* **Verifies** policy properties at compile time with a `VERIFY`
  block and CI-friendly exit-code semantics.
* **Diagnoses** six classes of reward misspecification with a
  `--diagnose` autopsy engine, and proposes minimum-magnitude
  repairs with `--repair`.
* **Analyzes execution logs**: a separate `mdp_autopsy` binary
  reads logs from a deployed run and reports model-vs-reality
  divergence.

---

## Quick start

### Build

```bash
git clone <your-fork-url> mdp-dsl
cd mdp-dsl
./build.sh                # Linux / macOS
# or
BUILD.bat                 # Windows
```

The build produces two binaries: `bin/mdp_compiler` and
`bin/mdp_autopsy`. Each builds with a single `g++ -O2 -std=c++17`
invocation; no Makefile, no CMake.

### Hello, MDP

Save the following as `hello.mdp`:

```
STATE: A
STATE: B
STATE: Goal

ACTION: Go
ACTION: Stay

TRANSITION: A    Go   B    1.0
TRANSITION: B    Go   Goal 1.0
TRANSITION: Goal Go   Goal 1.0
TRANSITION: A    Stay A    1.0
TRANSITION: B    Stay B    1.0
TRANSITION: Goal Stay Goal 1.0

REWARD: Goal 10.0
ACTION_REWARD: A Go -0.1
ACTION_REWARD: B Go -0.1

DISCOUNT: 0.9
SOLVER:   vi

VERIFY: pi(A) == Go
VERIFY: pi(B) == Go
VERIFY: V(Goal) > V(A)
```

Run it:

```bash
./bin/mdp_compiler hello.mdp
```

You'll see the solved value function, the policy, and `VERIFY PASS`
for each assertion. Exit code is 0 on success, 1 if any VERIFY
fails.

### Try the Tiger POMDP

```bash
./bin/mdp_compiler examples/tiger.mdp
```

This is the canonical POMDP benchmark. PBVI solves it in roughly
100 backups; both `VERIFY` assertions pass.

### Try the portfolio optimizer

```bash
python3 portfolio/portfolio_server.py
# Then visit http://localhost:8421
```

The portfolio dashboard demonstrates the HMM-to-MDP bridge end to
end. A demo mode is available if you don't want to run the server.

---

## Language reference

The IEEE-format paper in `paper/` contains the complete language
reference (Section II). Brief summary:

```
STATE: <name>                              # declare a state
ACTION: <name>                             # declare an action
TRANSITION: <s> <a> <s'> <prob>            # T(s, a, s') = p
REWARD: <s> <value>                        # R(s)
ACTION_REWARD: <s> <a> <value>             # R(s, a)
DISCOUNT: <gamma>                          # gamma in [0, 1)
SOLVER: vi | pi | ql | pbvi                # solver backend

# POMDP extensions:
OBSERVATION: <name>
OBSERVE_PROB: <s> <a> <o> <prob>           # O(s, a, o) = p
INITIAL_BELIEF: <s> <prob>
BELIEF_STATE: <name> <s>:<p> <s>:<p> ...
ALPHA_VECTORS:   <int>                     # PBVI parameter, default 15
PBVI_ITERATIONS: <int>                     # default 100

# HMM bridge:
HMM_STATES: <name> <name> ...
EMISSION_TYPE: gaussian | discrete
FIT_HMM: <csv_path> iterations=<int>
HMM_SEED: <int>
BRIDGE: hmm -> mdp

# Verification (exit-code semantics, suitable for CI):
VERIFY: pi(<s_or_belief>) == <action>
VERIFY: pi(<s_or_belief>) != <action>
VERIFY: V(<s_or_belief>) > <number>
VERIFY: V(<s_or_belief>) < V(<s_or_belief>)

# Comments start with #
# Keywords are case-insensitive
```

---

## CLI reference

### `mdp_compiler`

```
mdp_compiler <file.mdp>                    # parse, validate, solve, verify
mdp_compiler <file.mdp> --json             # emit JSON instead of text
mdp_compiler <file.mdp> --diagnose         # run full autopsy (6 failure classes)
mdp_compiler <file.mdp> --diagnose=reward  # targeted: reward issues only
mdp_compiler <file.mdp> --fragility        # policy fragility analysis
mdp_compiler <file.mdp> --repair "pi(s)==a"  # find minimum fix for assertion
mdp_compiler <file.mdp> --show-hmm         # print learned HMM parameters
mdp_compiler <file.mdp> --animate          # ASCII visualizer of solver convergence
mdp_compiler <file.mdp> --html out.html    # HTML report
```

Exit codes: 0 if all `VERIFY` assertions pass, 1 otherwise.
`--diagnose` exits 1 if any issues are found.

### `mdp_autopsy`

```
mdp_autopsy <model.mdp> <run.log>          # diagnose model-vs-reality divergence
mdp_autopsy <model.mdp> <run.log> --json   # machine-readable output
mdp_autopsy <model.mdp> <run.log> --report-only  # skip backwards solver
```

Exit codes: 0 = MODEL_ACCURATE, 1 = MODEL_OPTIMISTIC, 2 =
MODEL_DANGEROUS, 3 = parse error.

---

## Repository layout

```
mdp-dsl/
  README.md
  LICENSE                          MIT
  CONTRIBUTING.md
  RESEARCH_RESERVED.md             open research directions
  build.sh / BUILD.bat             one-command build

  src/
    mdp_compiler.cpp               main compiler (~4,130 lines)
    mdp_autopsy.cpp                backwards-solver binary
    visualizer.{hpp,cpp}           terminal animation
    html_output.{hpp,cpp}          HTML report generator

  tests/
    test_framework.hpp             tiny test framework
    run_all_tests.cpp              174 tests, 297 assertions

  examples/
    tiger.mdp                      canonical POMDP benchmark
    robot_nav.mdp                  grid-world navigation
    portfolio.mdp                  HMM-bridged portfolio
    autopsy_demo/                  one .mdp per failure class

  portfolio/
    portfolio_dashboard.html       dark-themed dashboard (no npm)
    portfolio_server.py            stdlib HTTP server
    optimizer.py                   NSE/BSE optimizer pipeline
    price_fetcher.py               yfinance with 15-min cache

  paper/
    mdp_dsl_paper.pdf              IEEE-format paper

  data/
    portfolio_returns.csv          sample HMM training data
```

---

## Test suite

```bash
cd tests
g++ -std=c++17 -O2 -DMDP_TESTING_MODE -o run_tests run_all_tests.cpp
./run_tests
```

You should see `297/297 assertions passed`.

| Group                                       | Tests | Assertions |
| ------------------------------------------- | ----: | ---------: |
| Baseline (parser, validator, VI, PI, QL)    |    94 |        142 |
| POMDP / PBVI / belief update                |    21 |         47 |
| HMM / Baum-Welch / bridge                   |    15 |         36 |
| Autopsy engine (6 classes + REPAIR)         |    28 |         43 |
| Backwards solver / log parser               |    16 |         29 |
| **Total**                                   | **174** | **297** |

---

## Honest scope notes

A few things to know up front, so they aren't surprises later:

* **The backwards solver in `mdp_autopsy`** searches along the
  empirical direction, not freely over the L1 simplex. A free L1
  minimization would need a QP solver, which would violate the
  zero-dependency goal. The approximation is documented in the
  autopsy output itself.

* **The REPAIR command** only searches single-parameter
  perturbations. Multi-parameter joint repair is future work.

* **PBVI is a teaching-grade implementation**, not a SARSOP-class
  solver. Tiger converges correctly; larger POMDPs may need more
  belief points or longer runs.

* **AI-assisted implementation**: the post-coursework extensions
  (POMDP/PBVI, HMM bridge, portfolio, autopsy engine) were
  implemented with help from Claude AI under my direction. The
  v2.0 baseline (VI / PI / Q-Learning) and all architectural
  decisions are mine. The paper's Author Contribution Statement
  has full disclosure.

---

## Future direction

The current implementation is a compiler. The longer-term goal is
to grow it into a small IDE / SDE for RL specification: a workspace
where people can write a model, see solver and autopsy output
interactively, iterate, and version the spec as a real artifact.

If you'd like to help with any of that, or if you find a bug, see
[CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are
welcome.

---

## Paper and citation

The full IEEE-format paper is in `paper/`. If you reference this
work, please cite:

```bibtex
@misc{rajani2026mdpdsl,
  author = {Charvit Rajani},
  title  = {MDP-DSL: A Compiled Domain-Specific Language for
            Sequential Decision-Making with Formal Verification
            and Failure Diagnosis},
  year   = {2026},
  note   = {Indian Institute of Technology Guwahati}
}
```

## License

MIT. See [LICENSE](LICENSE).

Note: please read [RESEARCH_RESERVED.md](RESEARCH_RESERVED.md)
before working on certain research directions listed there.

## Acknowledgments

The v2.0 baseline (Value Iteration, Policy Iteration, Q-Learning,
the VERIFY block, the GRID macro, the original five-phase pipeline)
was completed under the supervision of Prof. Shyamanta M. Hazarika,
IIT Guwahati, as the author's DA 221M Artificial Intelligence
minor-course term project.
