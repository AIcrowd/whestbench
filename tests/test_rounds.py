"""Per-round configs, and the doc table that must not drift from them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from whestbench.budget import (
    CURRENT_ROUND,
    DEFAULT_FLOP_BUDGET,
    DEFAULT_LAMBDA_FLOPS_PER_SECOND,
    PHASE1_FLOP_BUDGET,
    PHASE1_LAMBDA_FLOPS_PER_SECOND,
    PHASE1_ROUND,
    PHASE2_ROUND,
    ROUNDS,
    WARMUP_FLOP_BUDGET,
    WARMUP_ROUND,
)

_DOC = Path(__file__).resolve().parent.parent / "docs" / "reference" / "rounds.md"


def test_every_round_is_registered_by_its_dataset_tag():
    # Keyed by tag so a dataset revision maps straight onto its rulebook.
    assert set(ROUNDS) == {"v1-warmup", "v1-phase1", "v2-phase2"}
    for tag, round_ in ROUNDS.items():
        assert round_.tag == tag


def test_current_round_is_phase_2():
    assert CURRENT_ROUND is PHASE2_ROUND
    assert CURRENT_ROUND.tag == "v2-phase2"


@pytest.mark.parametrize(
    "round_,width,depth,budget",
    [
        (WARMUP_ROUND, 256, 8, 68_000_000_000),
        (PHASE1_ROUND, 256, 32, 272_000_000_000),
        (PHASE2_ROUND, 1024, 16, 2**41),
    ],
)
def test_round_shape_and_budget_match_the_baked_datasets(round_, width, depth, budget):
    """Pins the values against the published datasets.

    Each was read from that tag's ``metadata.json`` on
    ``aicrowd/arc-whestbench-public-2026``; if one of these ever needs changing,
    the dataset is the authority, not this file.
    """
    assert (round_.width, round_.depth) == (width, depth)
    assert round_.flop_budget == budget
    assert round_.n_samples == 1_000_000_000


def test_warmup_and_phase1_priced_residual_time():
    # Both rounds converted residual seconds to FLOPs at 1e11 and gated nothing.
    for round_ in (WARMUP_ROUND, PHASE1_ROUND):
        assert round_.residual_mode == "priced"
        assert round_.lambda_flops_per_second == 1e11
        assert round_.residual_wall_time_limit_s is None
        assert round_.wall_time_limit_s == 60.0


def test_phase2_gates_residual_time_instead_of_pricing_it():
    assert PHASE2_ROUND.residual_mode == "gated"
    assert PHASE2_ROUND.lambda_flops_per_second == 0.0, (
        "phase 2 deprecates residual pricing; a non-zero rate here would make "
        "C != F and reintroduce the second currency the gate removed"
    )
    assert PHASE2_ROUND.residual_wall_time_limit_s == 0.4
    assert PHASE2_ROUND.wall_time_limit_s == 120.0


def test_the_deprecated_rate_is_still_available_for_rescoring():
    # Deprecated, not removed -- a v1-* round cannot be re-scored without it.
    assert PHASE1_LAMBDA_FLOPS_PER_SECOND == 1e11


def test_module_defaults_are_derived_from_the_current_round():
    """The defaults must not be able to disagree with CURRENT_ROUND.

    They are derived rather than restated precisely so advancing a phase is one
    edit. If someone re-inlines a literal, this fails.
    """
    assert DEFAULT_FLOP_BUDGET == CURRENT_ROUND.flop_budget
    assert DEFAULT_LAMBDA_FLOPS_PER_SECOND == CURRENT_ROUND.lambda_flops_per_second
    assert PHASE1_FLOP_BUDGET == PHASE1_ROUND.flop_budget
    assert WARMUP_FLOP_BUDGET == WARMUP_ROUND.flop_budget


def test_rounds_are_immutable():
    # A round is a historical record; mutating one would rewrite the past.
    with pytest.raises(Exception):
        PHASE1_ROUND.flop_budget = 1  # type: ignore[misc]


# --- the doc table must not drift -------------------------------------------


def test_the_doc_lists_every_round():
    doc = _DOC.read_text()
    for tag in ROUNDS:
        assert f"`{tag}`" in doc, f"docs/reference/rounds.md does not mention {tag}"


@pytest.mark.parametrize("round_", list(ROUNDS.values()), ids=lambda r: r.tag)
def test_the_doc_states_each_rounds_shape(round_):
    doc = _DOC.read_text()
    assert re.search(rf"{round_.width}\s*×\s*{round_.depth}\b", doc), (
        f"docs/reference/rounds.md does not state {round_.tag}'s "
        f"{round_.width}x{round_.depth} shape"
    )


def test_the_doc_marks_phase2_residual_pricing_as_deprecated():
    # The wording participants rely on to know the rate no longer applies.
    doc = _DOC.read_text().lower()
    assert "deprecated" in doc
    assert "gated" in doc


def test_the_doc_pins_the_current_dataset_tag_not_an_older_one():
    """The doc's copy-pasteable command must name the round being graded.

    Quoting a retired tag in the pinning advice is exactly the failure the advice
    is warning about.
    """
    doc = _DOC.read_text()
    assert f"@{CURRENT_ROUND.tag}" in doc
    pin_section = doc.split("## Pinning the dataset")[-1]
    for stale in (t for t in ROUNDS if t != CURRENT_ROUND.tag):
        assert f"@{stale}" not in pin_section, f"the pinning example quotes the retired tag {stale}"
