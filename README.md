# BellmanDSL-Compiler-
A C++17 DSL compiler for modeling and solving MDPs/POMDPs with AST validation, formal verification, and support for Value/Policy Iteration, PBVI, and Q-Learning.

## What Is This?

MDP-DSL is a custom programming language and compiler for defining and solving **Markov Decision Processes** (MDPs) — the mathematical framework behind AI decision-making. Instead of writing error-prone C++ with manual array indexing, you describe your MDP in a simple `.mdp` text file and the compiler does the rest.

**The pipeline:**

```
.mdp file → [Parser] → [Validator] → [Solver] → Optimal Policy
```

## Why Use a DSL?

| Problem with raw C++ | DSL Solution |
|---|---|
| Probabilities might not sum to 1.0 | Validator catches this automatically |
| Typo in a state name → silent bug | Validator flags undeclared states |
| Must rewrite solver for each new MDP | Write a new `.mdp` file, reuse compiler |
| Data and algorithm tangled together | Complete separation of concerns |

## Quick Start

### 1. Compile

```bash
cd src/
g++ -std=c++17 -O2 -Wall -Wextra -o mdp_compiler mdp_compiler.cpp
```

### 2. Write a `.mdp` file

```
STATE: Start
STATE: Goal

ACTION: Go

TRANSITION: Start Go Goal 0.9
TRANSITION: Start Go Start 0.1
TRANSITION: Goal Go Goal 1.0

REWARD: Start 0.0
REWARD: Goal 10.0

DISCOUNT: 0.9
```

### 3. Run

```bash
./mdp_compiler your_file.mdp
```

The compiler will parse, validate (7 automatic checks), and solve the MDP, printing the optimal value function V*(s) and policy π*(s).

## The `.mdp` Grammar

| Keyword | Format | Example |
|---|---|---|
| `STATE:` | `STATE: <name>` | `STATE: Start` |
| `ACTION:` | `ACTION: <name>` | `ACTION: MoveRight` |
| `TRANSITION:` | `TRANSITION: <src> <act> <dst> <prob>` | `TRANSITION: Start Go Goal 0.9` |
| `REWARD:` | `REWARD: <state> <value>` | `REWARD: Goal 10.0` |
| `DISCOUNT:` | `DISCOUNT: <gamma>` | `DISCOUNT: 0.9` |

- Lines starting with `#` are comments
- Blank lines are ignored
- Keywords are case-insensitive; state/action names are case-sensitive

## Validation Checks (Phase 2)

The compiler automatically catches these errors before solving:

1. **Undeclared source state** in transitions
2. **Undeclared destination state** in transitions
3. **Undeclared action** in transitions
4. **Probability sums ≠ 1.0** (within ε = 1e-9 tolerance)
5. **Missing rewards** (warning)
6. **Orphan states** — declared but never used (warning)
7. **Undeclared reward states** — reward references non-existent state

## Solver (Phase 3)

Implements **Value Iteration** with the Bellman equation:

```
V(s) = R(s) + γ · max_a Σ T(s,a,s') · V(s')
```

- Convergence threshold: θ = 1e-9
- Maximum iterations: 10,000
- Acceleration structure: transition index for O(|T|) per iteration

## Example Files

| File | Description |
|---|---|
| `examples/gridworld_simple.mdp` | 4-state test MDP |
| `examples/gridworld_4x3.mdp` | Classic 4×3 Russell & Norvig GridWorld |
| `examples/invalid_mdp.mdp` | Deliberately broken MDP (tests validator) |
| `examples/edge_cases.mdp` | Parser edge case tests |

## Project Structure

```
mdp_dsl/
├── src/
│   ├── parser.cpp              # Phase 1 standalone parser
│   ├── mdp_compiler.cpp        # Unified pipeline (Phase 1+2+3)
│   └── gridworld_hardcoded.cpp # Phase 4 hardcoded comparison
├── examples/
│   ├── gridworld_simple.mdp    # 4-state test
│   ├── gridworld_4x3.mdp      # 4×3 case study
│   ├── invalid_mdp.mdp         # Validator tests
│   └── edge_cases.mdp          # Parser tests
├── docs/
│   ├── phase1_report.pdf       # Parser Engine
│   ├── phase2_report.pdf       # Semantic Validator
│   ├── phase3_report.pdf       # Value Iteration Solver
│   ├── phase4_report.pdf       # GridWorld Case Study
│   └── phase5_report.pdf       # Final Report
└── README.md
```

## Documentation

Each phase has a self-contained LaTeX report that defines every concept from scratch. A reader with zero prior knowledge of compilers or AI can learn the entire project by reading the reports in order.

## Requirements

- C++ compiler supporting C++17 (g++ 7+, clang++ 5+)
- No external libraries required
