# Team 2 — Project 2

## Overview
**Project 2 — Learning HMM Parameters with the Baum–Welch Algorithm**

Work on a training dataset that contains only observation sequences, with no hidden-state labels. Use the Baum–Welch (learning) algorithm to estimate the HMM parameters $P$, $E$, and $\pi$ from the observations alone.

## Document
- Detailed requirements are in [project2.pdf](./project2.pdf).

## Key Requirements:
- **R1:** Observation dataset description and preprocessing.
- **R2:** Model design (number of hidden states $N$).
- **R3:** Initialization of $P$, $E$, and $\pi$.
- **R4:** Baum–Welch implementation in Python (forward/backward, $\gamma$, $\xi$, re-estimation formulas).
- **R5:** Convergence analysis and log-likelihood plot.
- **R6:** Final learned parameters ($P$, $E$, $\pi$).
- **R7:** Comparison of multiple initializations.
- **R8:** Sequence decoding with Viterbi or sequence likelihood.
