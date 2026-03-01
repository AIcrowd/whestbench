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
