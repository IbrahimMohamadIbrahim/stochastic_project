# Team 1 — Project 1

## Overview
**Project 1 — Application of a Markov Model or a Hidden Markov Model**

Choose a real-life problem, model it as a Markov chain or as a Hidden Markov Model (HMM), build its matrices, and analyze it using the methods covered in the course.

## Document
- Detailed requirements are in [project1.pdf](./project1.pdf).

## Key Requirements:
- **R1:** Problem and model definition.
- **R2:** Transition matrix $P$ and state-transition diagram.
- **R3:** $n$-step behavior ($\pi \cdot P^n$).
- **R4:** Steady state analysis ($\pi = \pi \cdot P$).
- **R5:** Absorption analysis ($N = (I - Q)^{-1}$, $B = N \cdot R$, $t = N \cdot 1$).
- **R6–R8 (if HMM):** Parameters, Forward algorithm (Evaluation), Viterbi algorithm (Decoding).
- **Implementation:** Python.
