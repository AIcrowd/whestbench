# Participant Docs IA + Professional Polish Design

## 1. Goal

Deliver a polished GitHub Markdown documentation experience for first-time challenge participants that maximizes:

- clarity of onboarding,
- perceived professionalism and trust,
- fast transition from clone to first successful local run.

The design explicitly avoids Mermaid-first presentation and prefers concise, reliable text formatting.

## 2. Audience And Scope

Primary audience:

- participants already familiar with Python tooling (`uv`, CLI workflows, virtual environments).

In-scope surfaces:

- `README.md`
- participant-facing `docs/*`
- `tools/circuit-explorer/README.md`

Out of scope for public starter-kit docs:

- any internal workflow documentation intended only for maintainers.

## 3. Documentation Architecture (FastAPI-Inspired)

Adopt a strict top-level docs taxonomy:

- `docs/getting-started/`
- `docs/concepts/`
- `docs/how-to/`
- `docs/reference/`
- `docs/troubleshooting/`

Core IA rules:

- `README.md` is the front door, not the full manual.
- `docs/index.md` is the canonical table of contents.
- Each page has a clear type (`Concept`, `How-To`, `Reference`) and purpose.
- Participants should always have a clear next page to read.

## 4. README Strategy

`README.md` becomes a concise entrypoint with:

- one-screen challenge overview,
- shortest happy-path commands (`install -> init -> validate -> run -> package`),
- a structured docs map by category,
- explicit notes on current platform boundaries without TODO-style phrasing.

`README.md` should avoid:

- long conceptual deep dives,
- exhaustive command/flag references,
- duplicated content already covered in guides.

## 5. Content Model And Writing Standard

### 5.1 Getting Started

Each page includes:

- prerequisites,
- exact commands,
- expected outcomes,
- next-step links.

### 5.2 Concepts

Each page includes:

- challenge/problem framing,
- scoring intuition and formal vocabulary,
- constraints and common failure modes.

Minimal operational commands.

### 5.3 How-To

Each page includes:

- task-focused procedures,
- copy-paste commands,
- common variants and decision points.

### 5.4 Reference

Each page includes:

- authoritative CLI/contract details,
- concise flag and parameter semantics,
- artifact format expectations.

### 5.5 Troubleshooting

Each page includes:

- symptom,
- likely cause,
- concrete fix.

## 6. Diagram And Visual Policy

Initial policy for this pass:

- remove Mermaid diagrams from participant docs,
- replace with structured text flows and compact callouts.

Rationale:

- reduce rendering inconsistencies,
- improve portability across GitHub markdown renderers,
- keep docs easy to maintain in plain text.

## 7. Sanitization-Aware Constraints

Public participant docs must be valid after sanitization and must not depend on internal paths removed during publish.

Known sanitized paths include:

- `.aicrowd/`
- `docs/context/`
- `docs/plans/`
- `CHALLENGE-CONTEXT.md`
- `docs/development/worktrees-and-cli.md`
- `tools/circuit-explorer/STYLE_GUIDE.md`

Design rule:

- no participant-facing docs may link to or mention sanitized/internal locations.

## 8. Migration Strategy

1. Build new docs category directories and `docs/index.md`.
2. Re-map existing guides by document intent, not filename parity.
3. Rewrite `README.md` into front-door format.
4. Remove Mermaid blocks and convert to textual flow descriptions.
5. Update `tools/circuit-explorer/README.md` messaging to align with participant docs.
6. Preserve compatibility where useful via thin moved-page stubs.

## 9. Validation And Quality Gates

Required checks for this redesign:

- markdown link integrity across `README.md`, `docs/*`, and explorer readme,
- CLI command examples match actual parser behavior,
- docs quality tests updated to enforce the new IA and onboarding guarantees,
- no participant-facing references to sanitized/internal paths,
- manual onboarding smoke path succeeds from clean clone:
  - install tooling
  - `cestim init`
  - `cestim validate`
  - `cestim run`

## 10. Risks And Mitigations

Risk: broken external links after renames.
Mitigation: add short redirect/moved pages for high-traffic old paths where needed.

Risk: polished prose but weaker precision.
Mitigation: keep strict reference pages for contracts and flags.

Risk: mixed positioning of Circuit Explorer.
Mitigation: unify wording so explorer is clearly optional, educational, and non-submission.

## 11. Success Criteria

The redesign is successful when:

- first-time participants can find the right page quickly without guessing,
- docs feel cohesive and production-ready,
- onboarding flow runs cleanly with provided commands,
- no internal/sanitized references leak into participant-facing markdown.
