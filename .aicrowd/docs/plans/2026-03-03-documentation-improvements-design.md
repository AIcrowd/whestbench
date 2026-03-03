# Documentation Improvements — Progressive Disclosure Rewrite

## Goal

Improve the participant-facing documentation so it serves both ML/TCS researchers and competitive programmers. The current docs are almost entirely procedural ("run this command, emit this shape") and lack the research framing, technical intuition, and problem explanation needed to make the challenge accessible and motivating.

## Approach

**Progressive Disclosure** — every page keeps its existing TL;DR and contract specs (good for competitive programmers), but gains substantive new sections below that provide the *why* and *intuition* for researchers. No restructuring; improve existing pages in-place.

## Pages to Change

### README.md

- Add opening paragraph framing mechanistic estimation before the quickstart
- Expand "Why this challenge matters" from 2 sentences into a proper paragraph
- Add a "practical goal" callout for competitive programmers

### docs/concepts/problem-setup.md

- Add "What is a circuit?" section with gate equation and layered architecture
- Add "Why depth makes this hard" section on correlation accumulation
- Add "What is mechanistic estimation?" definition
- Add "The sampling baseline" section

### docs/concepts/scoring-model.md

- Rewrite "High-level flow" with clearer language
- Add "Budget-adaptivity" section
- Add "The sampling comparison" intuition section
- Rewrite "Runtime rules" in plain English
- Add "What a good score looks like" micro-section

### docs/index.md

- Update section descriptions to match improved content

## Constraints

- Scoring details kept intentionally vague (broad idea, not exact formulas)
- No new pages — in-place improvements only
- Don't touch how-to, reference, troubleshooting, or getting-started docs
