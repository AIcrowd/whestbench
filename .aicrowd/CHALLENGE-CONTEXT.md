# Challenge Call: AIcrowd Challenge Proposal

## Challenge Title

**Mechanistic Estimation for Random MLPs**

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
* **Updated At:** March 19, 2026

---

## Challenge Description

This is a theoretical computer science challenge from the Alignment Research Center (ARC). It is based on the idea of producing **"mechanistic estimates"** for the loss of a neural network, as discussed in the following blog posts:

* [https://www.alignment.org/blog/competing-with-sampling/](https://www.alignment.org/blog/competing-with-sampling/)
* [https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

The challenge involves **random Multi-Layer Perceptrons (MLPs)** with ReLU activations.

Participants will design an algorithm that:

1. Accepts a description of a random MLP (width, depth, and weight matrices).
2. Outputs per-layer estimates of expected neuron activations when the network is fed Gaussian random inputs.

The MLPs are generated randomly with He-initialized weight matrices, producing **deep networks with complex activation patterns**.

---

## Motivation

We are excited to run this challenge to spur research on mechanistic estimation.

Random MLPs provide a strong testing ground where cleverly designed mechanistic estimates can perform well, and where there is rich structural behavior to exploit through ReLU activations and weight matrix analysis.

The natural baseline algorithm is to:

* Run the MLP many times on random Gaussian inputs.
* Take the empirical mean of activations at each layer.

However, we have already designed simple mechanistic algorithms (such as mean propagation and covariance propagation) that outperform this baseline.

We expect participants to:

* Rediscover these approaches.
* Explore a wide space of more sophisticated ideas.
* Help convey our research to a wider audience.
* Develop novel techniques of direct interest to our research agenda.

---

## Timeline

We will provide:

* A description of the procedure used to generate MLPs (He initialization, ReLU activations, Gaussian inputs).
* A toolkit (`nestim`) for generating MLPs, evaluating estimators, and packaging submissions.

Participants will also be able to generate unlimited data themselves using the provided generation procedure.

Participants must submit an algorithm that estimates per-layer expected neuron activations for a given MLP under a compute budget.

---

## Evaluation Criteria

Submissions will be evaluated on held-out MLPs using:

* **Mean Squared Error (MSE)** of predicted vs actual per-layer means
* Subject to a **runtime constraint** (compute budget) on the algorithm
* A scoring model that measures efficiency relative to a sampling baseline — can the estimator match sampling's accuracy in less time?

---

## Organizers Bio

**Paul Christiano** — Founder of ARC, former researcher at OpenAI. Expert in alignment research.

**Jacob Hilton** — Researcher at ARC, formerly at OpenAI. Expert in theoretical ML.

---

## MLP Construction

Each random MLP is defined by:

* **Width** (`n`): number of neurons per layer (default: 256)
* **Depth** (`d`): number of layers (default: 16)
* **Weight matrices**: `d` matrices of shape `(n, n)`, He-initialized: `W ~ N(0, 2/n)`
* **Activation**: ReLU (`max(0, x)`) applied after each layer
* **Input distribution**: `x ~ N(0, 1)` of shape `(n,)`

Forward pass: `x_{l+1} = ReLU(x_l @ W_l)` for each layer `l`.

The estimator's goal is to predict `E[x_l]` for each layer `l` — the expected activation vector when averaging over random inputs.

---

## Key Technical Properties

* At shallow depth, simple moment propagation (tracking means through ReLU) works well.
* At greater depth, correlations between neurons accumulate and break naive independence assumptions.
* The challenge difficulty scales with depth — deeper networks require more sophisticated structural analysis.
* The compute budget creates a time-accuracy tradeoff: sampling is always correct given enough time, but structural methods can converge faster.

---

## Platform

* Starter kit repository with `nestim` CLI
* Local evaluation, validation, and packaging tools
* Network Explorer visualization tool for interactive debugging
* Pre-built example estimators (random, mean propagation, covariance propagation, combined)
