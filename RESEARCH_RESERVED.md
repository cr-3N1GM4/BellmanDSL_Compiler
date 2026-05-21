# Research Directions

This file lists open research directions that are part of the
broader vision behind MDP-DSL. They are documented here partly as a
roadmap for the project and partly to publicly timestamp them as
intended future work by the author.

Contributions to the core compiler (bug fixes, new solver backends,
new semantic checks, better diagnostics, IDE/SDE integration, and
language extensions that don't conflict with the directions listed
below) are welcome under the standard MIT terms in `LICENSE`.

If you find any of these directions interesting and want to
collaborate, please open an issue or reach out before starting
independent work. The author's intent is to pursue each as a
research publication; co-authorship will be credited appropriately
where contributions are substantial.

---

## 1. Inverse MDP Compilation ("Policy Telepathy")

Given an observed sequence of state-action pairs
`[(s_1, a_1), (s_2, a_2), ..., (s_n, a_n)]` where each `a_t` is
believed to be `pi*(s_t)` for some unknown optimal policy `pi*`,
algebraically recover the MDP structure that produced it.

Formal statement:

```
Given:    O = [(s_t, a_t)]_{t=1..n}
Find:     M* = argmin_M  || pi_M - pi_observed ||
Subject:  M is a well-formed MDP-DSL program.
```

This is conceptually distinct from statistical Inverse RL (Ng &
Russell, 2000) in that it aims to recover the algebraic structure
of M, not just a reward function consistent with the observations.

Proposed surface syntax:

```
OBSERVE:
  state TrafficLight=Red    action=Wait
  state TrafficLight=Yellow action=Accelerate
INFER: MDP
AMBIGUITY_REPORT: on
```

Key open question: under what conditions on `O` is the inferred MDP
unique up to a known equivalence?

## 2. Discount-Cliff Cartography

The discount-cliff detector in the autopsy engine finds the single
cliff nearest the user's gamma. The full open question is: for a
given MDP, characterize the complete set of gamma values at which
policy phase transitions occur.

This is a piecewise-rational function of gamma; the combinatorial
structure (number of cliffs as a function of |S|, |A|, R, T) is not
well understood. A characterization would enable global stability
analysis of an MDP across all possible discount factors.

## 3. MDP Quines

Definition: an MDP Q is a quine if solving Q produces, as part of
its output, an MDP Q' such that `pi_Q = pi_{Q'}` and the
specification of Q' is bit-for-bit equal to the specification of Q.

Open questions:
* Does any finite MDP admit a quine construction?
* If yes, what is the minimum state count?
* Is the quine MDP unique up to isomorphism?

The `VERIFY` block is the natural primitive to formalize this:
`VERIFY: pi(s) == <output_of_solving_this_mdp_for_s>` is a
self-reference operator.

## 4. MDP Quine Chains

A directed cycle of MDPs `M_1 -> M_2 -> ... -> M_k -> M_1` where
the solved policy of `M_i` generates the source specification of
`M_{i+1}`. Existence and properties unknown.

## 5. Policy Metamorphosis

Continuous interpolation through MDP-space. Given two MDPs `M_0`
and `M_1` over the same state-action set, parameterize a family
`M_lambda` for `lambda in [0, 1]` and study the discrete
phase-transition structure of `pi*_lambda`. The discount cliff is a
1-dimensional special case.

## 6. Goedelian MDP

An MDP construction that encodes logical incompleteness, in the
sense that its `VERIFY` block contains an assertion neither
provable nor refutable by the compiler under standard solver
backends. Existence and minimal-size analogues unknown.

## 7. Policy Archaeology

Given black-box agent traces (only state-action pairs, no rewards,
no transitions), reconstruct the minimal MDP consistent with the
traces. Differs from inverse MDP compilation in that the
specification format is unconstrained; we ask for the smallest MDP
that could have produced the traces.

## 8. Universal Objective Autopsy

Generalize the six-class autopsy engine beyond MDPs to other
objective-function settings: deep learning loss functions,
genetic-algorithm fitness functions, optimization objectives in
constraint programming. Many failure modes (cycles, magnitude
imbalance, dead states, discount-cliff-like phase transitions) have
natural analogues in these domains.

## 9. POMDP-DSL (extension of the present work)

The PBVI solver included here is a teaching-grade implementation.
A full POMDP-focused language extension would add: HSVI-class
solver backends, finite-horizon POMDPs with proper terminal-belief
handling, partially-observable Markov games (two-player), and a
richer belief-property language for VERIFY (for example, "no policy
in `Gamma` selects `a` at any belief with `b(s) > 0.65`").

---

## Why this list is here

I think it's healthy to be explicit about which research questions
the author intends to pursue, rather than have them surface
implicitly later. If you find any of these interesting and want to
work on them, please reach out first; if you'd like to contribute
to the core compiler or to directions not on this list, please
just open a PR or issue.

The intent is collaborative, not exclusionary.
