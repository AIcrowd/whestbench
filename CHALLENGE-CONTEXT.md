# Challenge Call: AIcrowd Challenge Proposal

## Challenge Title

**Mechanistic Estimation for Boolean Circuits**

---

## Organization

**Alignment Research Center (ARC)**
Non-profit research organization conducting theoretical research aimed at aligning future machine learning systems with human interests.
Website: [https://www.alignment.org/](https://www.alignment.org/)

---

## Contact Information

* **Contact Name:** Jacob Hilton
* **Email:** [jacob@alignment.org](mailto:jacob@alignment.org)
* **Phone:** 415-230-9976

---

## Administrative Details

* **Created At:** February 10, 2026
* **Updated At:** February 10, 2026

---

## Challenge Description

This is a theoretical computer science challenge from the Alignment Research Center (ARC). It is based on the idea of producing **"mechanistic estimates"** for the loss of a neural network, as discussed in the following blog posts:

* [https://www.alignment.org/blog/competing-with-sampling/](https://www.alignment.org/blog/competing-with-sampling/)
* [https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

The challenge will involve **Boolean circuits** instead of neural networks.

Participants will design an algorithm that:

1. Accepts a description of a Boolean circuit.
2. Outputs the expected value of the output of the circuit (treating `TRUE` as 1 and `FALSE` as 0) on a uniformly random input.

The circuits will be generated randomly in a specific way designed to produce **deep circuits with complex behavior**.

---

## Motivation

We are excited to run this challenge to spur research on mechanistic estimation.

Boolean circuits provide a strong testing ground where cleverly designed mechanistic estimates can perform well, and where there is rich structural behavior to exploit.

The natural baseline algorithm is to:

* Run the circuit many times on random inputs.
* Take the empirical expectation.

However, we have already designed simple mechanistic algorithms (as yet unpublished) that outperform this baseline.

We expect participants to:

* Rediscover these approaches.
* Explore a wide space of more sophisticated ideas.
* Help convey our research to a wider audience.
* Develop novel techniques of direct interest to our research agenda.

---

## Timeline

We will provide:

* A description of the procedure used to generate circuits.
* Potentially, a dataset of example circuits generated using this procedure.

Participants will also be able to generate unlimited data themselves using the provided generation procedure.

Participants must submit an algorithm that estimates the expected value of the circuit on a uniformly random input.

---

## Evaluation Criteria

Submissions will be evaluated on held-out circuits using:

* **Mean Squared Error (MSE)**
* Subject to a **runtime constraint** on the algorithm

Evaluation may include:

* **In-distribution evaluation** — circuits generated using the same method.
* **Out-of-distribution evaluation** — circuits generated using variant methods.

---

## Organizers Bio

We are a non-profit organization founded in 2021.

Our research has been published at top machine learning and theoretical computer science conferences.

We have not previously run a public challenge.

---

## Prize & Logistics

* **Prize Pool:** $50,000 – $100,000
* Philanthropic funding secured to cover:

  * Prize money
  * Operational costs associated with running the contest

We would love to hop on a call to discuss further.



# Email Thread: AIcrowd × Alignment Research Center

**Subject:** Challenge Call Response – Mechanistic Estimation for Boolean Circuits
**Participants:** Sharada Mohanty (AIcrowd), Jacob Hilton (Alignment Research Center)
**Date Range:** Feb 10–12, 2026

---

# 📩 Full Email Thread (Verbatim Content Preserved)

---

## 📨 Email 1 — Challenge Call Submission

**From:** [no-reply@aicrowd.com](mailto:no-reply@aicrowd.com)
**To:** Sharada Mohanty
**Date:** Feb 10, 2026
**Subject:** New Challenge Call Submission

---

### Challenge Call Details

* **Call:** Challenges on AIcrowd
* **Contact Name:** Jacob Hilton
* **Organization:** Alignment Research Center
* **Phone:** 415-230-9976
* **Email:** [jacob@alignment.org](mailto:jacob@alignment.org)

---

### Proposal Title

**Mechanistic Estimation for Boolean Circuits**

---

### Proposal Summary (Exact Submission Text)

> This is a theoretical computer science challenge from the Alignment Research Center ([https://www.alignment.org/](https://www.alignment.org/)), a non-profit research organization conducting theoretical research aiming to align future machine learning systems with human interests.
>
> The challenge builds on the idea of producing "mechanistic estimates" for neural network loss, as discussed in:
>
> * [https://www.alignment.org/blog/competing-with-sampling/](https://www.alignment.org/blog/competing-with-sampling/)
> * [https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)
>
> Instead of neural networks, the challenge focuses on Boolean circuits.
>
> Participants must accept a description of a Boolean circuit and output the expected value of the circuit output (TRUE = 1, FALSE = 0), assuming uniformly random inputs.
>
> Circuits are randomly generated in a structured way to produce deep circuits with complex behavior.

---

## 📨 Email 2 — Sharada’s Initial Response

**From:** Sharada Mohanty
**To:** Jacob Hilton
**Date:** Feb 10, 2026

---

> Dear Jacob,
>
> Thank you for your interest in AIcrowd, and for the very interesting challenge proposal.
>
> It is also great to hear from you, as this closely aligns with a direction I have personally been very interested in — developing a series of meaningful, well-designed challenges around alignment and mechanistic interpretability on AIcrowd.
>
> Would you be available for a short call later this week (or early next week), for us to get a better understanding of the task and to discuss next steps?
>
> Before we speak, it would be helpful to clarify a few points:
>
> 1. Circuit Difficulty
>
>    * Do you already have a working notion of circuit difficulty?
>    * Is difficulty tied to depth, gate reuse, gate types?
>    * Does difficulty influence test set stratification?
> 2. Evaluation Design
>
>    * Runtime constraint + MSE on held-out circuits
>    * Should runtime budgets be per-circuit or per-batch?
>    * Should resource usage (CPU, memory) influence scoring?
> 3. Algorithm Scope
>
>    * Are per-instance reasoning methods preferred?
>    * Are offline-trained predictors acceptable?
>    * Should lightweight code review (e.g., LLM-as-judge) be considered?
> 4. Adversarial Optimization / Leaderboard Gaming
>
>    * Fitting generator parameters instead of solving circuits
>    * Embedding offline-trained predictors
>    * Exploiting floating-point quirks
>    * Guardrails to prevent unintended optimizations?
> 5. Timeline
>
>    * Potential NeurIPS Competition Track alignment?
>    * Earlier launch possible?
> 6. Blue Sky Award (Optional)
>
>    * Additional prize for extraordinary performance beyond expectations?
>
> Optional: If available, a minimal end-to-end reference implementation would help build shared intuition ahead of the call.
>
> Best,
> Mohanty

---

## 📨 Email 3 — Jacob’s Technical Clarifications

**From:** Jacob Hilton
**To:** Sharada Mohanty
**Date:** Feb 12, 2026

---

### Circuit Construction (As Described in Email)

> Proposed generator:
>
> * Start with n input wires (uniform random inputs)
> * For each of L layers:
>
>   * Replace n wires with new n wires
>   * Randomly pair wires
>   * Connect each pair with one of 16 possible 2-input gates
> * Final output: n output wires
>
> Goal: Estimate probability each output wire is on.
>
> Parameters:
>
> * n (width), e.g., 1024
> * L (depth), 1 to 256
> * Computational budget
> * Target MSE: 1e-3 to 1e-6

---

### Evaluation Philosophy (Exact Points Raised)

> * Constrain CPU and memory per circuit (per-batch possible)
> * Algorithms should take a resource budget as input and adapt
> * Performance improves sharply with compute
>
> Ideal properties:
>
> 1. Consistent computational environment
> 2. Transparent runtime feedback
> 3. ±10% compute buffer with linear score adjustment
> 4. CPU usage should factor into score (not just runtime cap)

---

### Algorithm Scope

> * Open to broad approaches
> * No LLM-based code review planned
> * Will red-team further before launch

---

### Adversarial Considerations

> * Expect best methods to be mechanistic
> * May run a trial hackathon to stress-test design

---

### Timeline

> * Desire to launch ASAP
> * Interested in compatibility with NeurIPS timeline

---

### Code Access

> Shared minimal repo:
> [https://github.com/alignment-research-center/circuit-estimation-mvp](https://github.com/alignment-research-center/circuit-estimation-mvp)

---

## 📨 Email 4 — Sharada’s Follow-Up & Platform Considerations

---

### Call Scheduling

> Proposed slot: Monday, 16th 9:00–10:00 PT

---

### Infrastructure & Evaluation Design Discussion

> Resource Enforcement Options:
>
> * Soft enforcement: Penalize overages
> * Hard enforcement: Throttle processes (container-level limits)
>
> Runtime Budget Adaptivity:
>
> * Reference: Data Purchasing Challenge (2022)
> * Participants often implemented regime-specific if-else logic rather than true adaptivity
> * Need careful budget exposure to incentivize genuine adaptive algorithms
>
> CPU Usage in Score:
>
> * Avoid unintended leaderboard pathologies
> * Avoid distorted optimization dynamics
>
> Environment Consistency:
>
> * All submissions run on predefined AWS instances
> * Detailed runtime and resource dashboards available
> * Participants can spin up identical AWS instances for validation

---

### Red-Teaming Proposal

> * Launch unlisted invite-only beta
> * Stress-test evaluation protocol
> * Validate leaderboard dynamics before public release

---

### Timeline Strategy Proposal

> Infrastructure readiness estimate: ~4 weeks + beta phase
>
> NeurIPS considerations:
>
> * Proposal deadline: mid-April
> * Acceptance: mid-May
> * Competition window: late June–October
>
> Suggested strategy:
>
> 1. Launch when ready
> 2. Submit to NeurIPS in parallel
> 3. If accepted, introduce additional NeurIPS-affiliated phase

---

## 📨 Final Acknowledgment

> Sharada acknowledged receipt of repo and planned deeper review ahead of call.

---

# End of Thread
