"""Verify the torch path's closed-form FLOP accounting matches flopscope's actual count.

This is the critical correctness gate: the torch path computes outside flopscope, so these
numbers are asserted rather than measured, and if flopscope changes its accounting upstream
this test must fail loudly rather than let the drift into an immutable dataset revision.

It checks the count PER OPERATION, not just the total. The earlier version compared
`flops_used` and then asserted `"calls" in bucket` / `"operations" in bucket` -- presence,
never content -- which is why a shipped `calls: 0` and `operations: {}` passed it for as long
as they existed. It also ran a single 100-sample chunk, so the chunk-size parameter was
never exercised at all.
"""

import json
import math

import pytest

from whestbench.dataset import create_dataset, load_dataset
from whestbench.dataset_torch import _synthesize_sampling_breakdown

flopscope = pytest.importorskip("flopscope")
from flopscope import numpy as fnp  # noqa: E402

from whestbench import simulation  # noqa: E402
from whestbench.domain import MLP  # noqa: E402
from whestbench.scoring import _normalize_sampling_budget_breakdown  # noqa: E402

NS = "sampling.sample_layer_statistics"


def _mlp(width, depth):
    import numpy as np

    rng = np.random.default_rng(0)
    return MLP(
        width=width,
        depth=depth,
        weights=[
            (rng.standard_normal((width, width)) / np.sqrt(width)).astype(np.float32)
            for _ in range(depth)
        ],
    )


def _flopscope_ops(mlp, n_samples):
    """What flopscope actually charges for one real sampling run."""
    with flopscope.BudgetContext(flop_budget=int(1e18), quiet=True) as ctx:
        with flopscope.namespace("sampling"):
            with flopscope.namespace("sample_layer_statistics"):
                simulation.sample_layer_statistics(mlp, n_samples, rng=fnp.random.default_rng(1))
    s = ctx.summary_dict(by_namespace=True)
    return s, s["by_namespace"][NS]


def _synth(width, depth, n_samples, **kw):
    return _synthesize_sampling_breakdown(
        width=width,
        depth=depth,
        n_samples=n_samples,
        chunk_size=simulation._pick_chunk_size(width),
        wall_time_s=0.0,
        **kw,
    )


# width 8 -> _pick_chunk_size 16384, so these are single-chunk;
# width 1024 -> 1024, giving k = 4 / 5 / 10 / 12 with two ragged finals.
# Both regimes matter: the chunk-dependent terms only separate across different k, and
# their SUM is k-independent, so a single-k test cannot see an error in the split.
@pytest.mark.parametrize(
    "width,depth,n_samples",
    [
        (8, 2, 100),  # single chunk — the original case
        (1024, 16, 4096),  # k=4, exact
        (1024, 16, 5000),  # k=5, ragged final chunk of 904
        (1024, 16, 10240),  # k=10, exact
        (1024, 16, 11267),  # k=12, ragged final chunk of 3
    ],
)
def test_closed_form_matches_flopscope_per_operation(width, depth, n_samples):
    mlp = _mlp(width, depth)
    summary, bucket = _flopscope_ops(mlp, n_samples)
    synth = _synth(width, depth, n_samples)
    sb = synth["by_namespace"][NS]

    assert synth["flops_used"] == summary["flops_used"], "total FLOP count drifted"
    assert sb["calls"] == bucket["calls"], "namespace call count drifted"
    assert set(sb["operations"]) == set(bucket["operations"]), (
        f"operation sets differ: flopscope-only="
        f"{sorted(set(bucket['operations']) - set(sb['operations']))}, synthesis-only="
        f"{sorted(set(sb['operations']) - set(bucket['operations']))}"
    )
    for op, real in bucket["operations"].items():
        got = sb["operations"][op]
        assert got["flop_cost"] == real["flop_cost"], f"{op}: flop_cost"
        assert got["calls"] == real["calls"], f"{op}: calls"


def test_a_one_row_final_chunk_is_billed_correctly():
    """The chunking the reduction model does not naturally cover.

    flopscope bills a degenerate 1-row sum(axis=0) a fixed per-call cost where the
    (n_c - 1) model predicts zero. Without the correction term the closed form is exactly
    2*(depth+1) low — and since `operations` sums to `flops_used`, that error would land in
    the headline number too, not just the breakdown.
    """
    width, depth = 1024, 16
    cs = simulation._pick_chunk_size(width)
    n = cs * 3 + 1  # final chunk holds exactly one row
    _, bucket = _flopscope_ops(_mlp(width, depth), n)
    synth = _synth(width, depth, n)

    assert (
        synth["by_namespace"][NS]["operations"]["sum"]["flop_cost"]
        == bucket["operations"]["sum"]["flop_cost"]
    )
    naive = 2 * (depth + 1) * (n - math.ceil(n / cs)) * width
    assert bucket["operations"]["sum"]["flop_cost"] - naive == 2 * (depth + 1) == 34


def test_operations_sum_to_the_namespace_total_and_use_flopscopes_key():
    """`flop_cost`, not `flops`: scoring._merge_operation_timing reads flop_cost, so any
    other key aggregates to zero with no error anywhere."""
    sb = _synth(1024, 16, 4096)["by_namespace"][NS]
    assert sb["operations"], "operations must not be empty"
    assert sb["calls"] > 0, "calls must not be zero"
    assert sum(o["flop_cost"] for o in sb["operations"].values()) == sb["flops_used"]
    assert sum(o["calls"] for o in sb["operations"].values()) == sb["calls"]
    for name, entry in sb["operations"].items():
        assert set(entry) == {
            "flop_cost",
            "calls",
            "flopscope_backend_time_s",
            "flopscope_overhead_time_s",
        }, name


def test_chunk_size_moves_the_split_but_not_the_total():
    """Why a wrong chunk_size hid for so long, expressed as a test.

    sum + add = 2(d+1)w(n-k) + 2(d+1)wk, so k cancels in the total and survives in the
    split. A synthesis that re-derives chunk_size from the CPU rule therefore produces a
    correct total over a breakdown describing a run that never happened.
    """
    totals, adds = set(), set()
    for cs in (1024, 8192, 65536, 524288):
        d = _synthesize_sampling_breakdown(
            width=1024, depth=16, n_samples=1_000_000, chunk_size=cs, wall_time_s=0.0
        )
        ops = d["by_namespace"][NS]["operations"]
        totals.add(d["flops_used"])
        adds.add(ops["add"]["flop_cost"])
    assert len(totals) == 1, "the total must not depend on chunking"
    assert len(adds) == 4, "the sum/add split MUST depend on chunking"


def test_chunk_size_is_required():
    """There is no correct default. The old code silently used the CPU path's rule."""
    with pytest.raises(TypeError):
        _synthesize_sampling_breakdown(  # type: ignore[call-arg]
            width=8, depth=2, n_samples=100, wall_time_s=0.0
        )


def test_unbudgeted_bake_records_a_consistent_budget():
    """A ground-truth bake has no cap. Recording budget == used keeps the identity exact
    without inventing one, and without a null that scoring's int() would choke on."""
    d = _synth(8, 2, 100, flop_budget=None)
    assert d["flop_budget"] == d["flops_used"]
    assert d["flops_remaining"] == 0
    assert d["flop_budget"] - d["flops_used"] == d["flops_remaining"]


def test_a_stated_budget_is_recorded_verbatim_and_the_remainder_is_not_clamped():
    """The old `max(0, budget - total)` reported a run that overshot as exactly exhausted —
    the more dangerous of the two readings."""
    d = _synth(1024, 16, 100_000, flop_budget=1_000)
    assert d["flop_budget"] == 1_000
    assert d["flops_remaining"] == 1_000 - d["flops_used"] < 0

    head = _synth(8, 2, 100, flop_budget=10**15)
    assert head["flops_remaining"] == 10**15 - head["flops_used"] > 0


def test_shape_parity_with_flopscopes_normalized_output(tmp_path):
    """The synthesized row must survive whestbench's own normalizer with numbers intact,
    and match what the flopscope-instrumented CPU path produces for the same shape."""
    width, depth, n_samples = 8, 2, 100
    out = create_dataset(
        n_mlps=2,
        n_samples=n_samples,
        width=width,
        depth=depth,
        mlp_seeds=[42000, 42001],
        output_path=tmp_path / "baseline",
    )
    actual = json.loads(load_dataset(out, split="public")["sampling_budget_breakdown"][0])
    synth = _synth(width, depth, n_samples)

    assert synth["flops_used"] == actual["flops_used"]
    assert set(synth) >= {
        "flop_budget",
        "flops_used",
        "flops_remaining",
        "wall_time_s",
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
        "residual_wall_time_s",
        "by_namespace",
    }
    assert synth["time_source"] == "bake"
    assert NS in synth["by_namespace"]

    norm = _normalize_sampling_budget_breakdown(synth)
    nb = norm["by_namespace"][NS]
    assert nb["flops_used"] == synth["flops_used"]
    assert nb["calls"] == synth["by_namespace"][NS]["calls"] > 0
    assert sum(o["flop_cost"] for o in nb["operations"].values()) == synth["flops_used"], (
        "operations aggregated to the wrong total — the symptom of using the key `flops` "
        "where scoring reads `flop_cost`"
    )


def test_synthesized_breakdown_carries_bake_tag_and_full_decomposition():
    """The torch path runs outside flopscope, so all bake wall clock is residual
    (wall = backend + overhead + residual, backend = overhead = 0)."""
    d = _synth(8, 2, 100, flop_budget=None)
    d["wall_time_s"] = d["wall_time_s"]
    assert d["flopscope_backend_time_s"] == 0.0
    assert d["flopscope_overhead_time_s"] == 0.0
    assert d["residual_wall_time_s"] == d["wall_time_s"]
    assert d["time_source"] == "bake"
