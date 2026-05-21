# Contributing to MDP-DSL

Thanks for your interest. This is a small project and I'd love help
with it.

## Quick start for contributors

1. Fork the repo and clone your fork.
2. Build: `./build.sh` (Linux/macOS) or `BUILD.bat` (Windows).
3. Run the test suite:
   ```
   cd tests
   g++ -std=c++17 -O2 -DMDP_TESTING_MODE -o run_tests run_all_tests.cpp
   ./run_tests
   ```
   You should see `297/297 assertions passed`.
4. Make your change.
5. Add a test (every behavior change should have a positive and
   negative test where it makes sense).
6. Open a PR.

## What kinds of contributions are most welcome

* **Bug reports.** Even small ones. Include a minimal `.mdp` file
  that triggers the issue, what you expected, and what you saw.
* **New examples.** Canonical MDP/POMDP problems beyond Tiger and
  GridWorld would be useful: machine maintenance, queueing, simple
  finance toys, classical inventory.
* **New solver backends.** SARSOP, HSVI, prioritized sweeping, all
  welcome. Keep the zero-dependency constraint in mind.
* **New semantic checks.** If you find a real-world misspecification
  that the validator doesn't catch, a new check is the right fix.
* **New autopsy detection classes.** The six classes are not a
  complete taxonomy.
* **Documentation improvements.** If something in the README or
  paper is confusing, a PR to clarify it is more useful than an
  issue.
* **Better error messages.** Anything that makes a compiler error
  more actionable.

## What to discuss before working on

* **Research directions listed in `RESEARCH_RESERVED.md`.** Please
  open an issue first if you want to work on any of these; I'd like
  to collaborate.
* **Major architectural changes.** If your change touches the
  five-phase pipeline structure or the AST shape, please open an
  issue first so we can talk through the design.

## Engineering constraints

These are part of the project's identity, not arbitrary:

* **Zero external C++ dependencies.** The compiler uses only the
  C++17 standard library. The Python pipeline may use yfinance,
  numpy, etc. The dashboard uses zero npm packages.
* **Single translation unit for `mdp_compiler.cpp`.** Build is one
  g++ invocation. If your change requires a new TU, please discuss
  first.
* **Backward compatibility.** Every existing `.mdp` file must
  continue to compile to byte-identical output. The CI runs the
  full example set; if a change breaks any example, it's blocked.
* **Tests.** Don't merge a change without tests. The bar is "the
  test would catch a regression if I broke this later."

## Style

C++ style is loose but consistent within the file. Match the
surrounding code. No clang-format config is enforced.

Python style is PEP 8 in spirit; nothing strict.

Comments: prefer brief explanatory comments at non-obvious
algorithmic steps to walls of docstring.

## A note on AI-assisted contributions

If you use AI tools (Claude, Copilot, ChatGPT, etc.) to help draft
contributions, that's fine. Please review what you submit; you're
responsible for it. I'd rather have a small, well-understood PR
than a large one whose authors can't explain it.

## Code of conduct

Be kind. Disagree with ideas rather than with people. If anything
in the project (or any interaction around it) makes you
uncomfortable, please open an issue or reach out directly.

## Maintainer

Charvit Rajani, IIT Guwahati. Issues and PRs are the best contact;
for research-direction collaborations (see `RESEARCH_RESERVED.md`),
please open an issue first.
