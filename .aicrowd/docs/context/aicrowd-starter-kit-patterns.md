# AIcrowd Starter Kit and Submission Patterns (External Research)

Last updated: 2026-03-01

This note captures public patterns from AIcrowd starter kits/docs that are relevant to shaping this repository into a participant-ready starter kit.

## What Appears Consistent Across Public Starter Kits

From public starter kit examples:

- Starter kits usually include a small set of required participant files.
- `aicrowd.json` is used to describe challenge metadata and constraints.
- A clear local run path is expected (for example baseline execution command).

Example public source (Music Demixing starter kit):

- Repository includes baseline code, docs, and an `aicrowd.json` at root.
- `aicrowd.json` snippet in public listing includes keys such as:
  - `challenge_id`
  - `authors`
  - `description`
  - `gpu`

Source:

- https://github.com/AIcrowd/music-demixing-challenge-starter-kit

## Explicit Required Files in One Public AIcrowd Challenge Doc

A public AIcrowd challenge page for food recognition lists required participant files:

- `README.md`
- `requirements.txt`
- `aicrowd.json`
- `train.py`
- `predict.py`

It also emphasizes that `README.md` should explain local training/inference commands.

Source:

- https://gitlab.aicrowd.com/aicrowd/challenges/food-recognition-benchmark-2022

## Submission Modality Signals (Needs Confirmation)

Public docs show mixed paradigms depending on challenge generation:

- Some docs/starter kits describe code-style submission workflows.
- The currently indexed AIcrowd CLI docs page titled "Creating a Submission" states:
  - git submissions are "not ready yet" (wording on that page), and
  - submission is done by uploading prediction artifacts via CLI.

Source:

- https://docs.aicrowd.com/cli/submissions

Interpretation:

- Submission mechanics on AIcrowd are challenge-dependent and may have evolved over time.
- For this ARC challenge, do not assume one global default until AIcrowd host-side submission mode is explicitly fixed.

## Practical Implications for This Repo

To prepare this repository for future starter-kit conversion, keep/add:

1. A strict participant contract document (inputs, outputs, runtime limits, required entrypoints).
2. A local evaluator command that mimics hosted scoring behavior.
3. A minimal metadata config stub (`aicrowd.json`-like) once challenge-side schema is confirmed.
4. Separate "toy benchmark logic" from "submission harness" code.

## Known Unknowns to Resolve with AIcrowd Team

- Final submission mode for this specific challenge (code package vs prediction artifact vs hybrid).
- Canonical schema for challenge metadata and run-time declarations.
- Exactly what local scripts AIcrowd injects into official starter kits during platform integration.

