"""Verify the torch path's closed-form FLOP count matches flopscope's actual count.

This is the critical correctness gate: if flopscope ever changes its accounting
model upstream, this test fails loudly and we update the synthesis formula.
"""

from whestbench.dataset import create_dataset, load_dataset
from whestbench.dataset_torch import _synthesize_sampling_breakdown


def test_closed_form_matches_flopscope_count(tmp_path) -> None:
    width = 8
    depth = 2
    n_samples = 100
    n_mlps = 2
    flop_budget = 1_000_000_000_000_000  # match what create_dataset uses internally

    out = create_dataset(
        n_mlps=n_mlps,
        n_samples=n_samples,
        width=width,
        depth=depth,
        seed=42,
        output_path=tmp_path / "flopscope_baseline.npz",
    )
    bundle = load_dataset(out)
    assert bundle.sampling_budget_breakdowns is not None
    actual = bundle.sampling_budget_breakdowns[0]
    actual_flops = actual["flops_used"]

    synthesized = _synthesize_sampling_breakdown(
        width=width,
        depth=depth,
        n_samples=n_samples,
        wall_time_s=0.0,
        flop_budget=flop_budget,
    )

    # Exact equality on the FLOP count — the load-bearing assertion.
    assert synthesized["flops_used"] == actual_flops, (
        f"Closed-form FLOP count {synthesized['flops_used']} does not match "
        f"flopscope's actual count {actual_flops}."
    )

    # Shape parity with flopscope's normalized output:
    assert set(synthesized.keys()) >= {
        "flop_budget",
        "flops_used",
        "flops_remaining",
        "wall_time_s",
        "flopscope_backend_time_s",
        "flopscope_overhead_time_s",
        "residual_wall_time_s",
        "by_namespace",
    }
    # by_namespace uses FLAT dot-notation keys (not nested):
    assert "sampling.sample_layer_statistics" in synthesized["by_namespace"]
    bucket = synthesized["by_namespace"]["sampling.sample_layer_statistics"]
    assert bucket["flops_used"] == actual_flops
    assert "calls" in bucket
    assert "operations" in bucket

    # Sanity check on flops_remaining
    assert synthesized["flops_remaining"] == flop_budget - actual_flops
