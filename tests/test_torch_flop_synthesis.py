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

    out = create_dataset(
        n_mlps=n_mlps,
        n_samples=n_samples,
        width=width,
        depth=depth,
        flop_budget=32,
        seed=42,
        output_path=tmp_path / "flopscope_baseline.npz",
    )
    bundle = load_dataset(out)
    assert bundle.sampling_budget_breakdowns is not None

    actual_flops = bundle.sampling_budget_breakdowns[0]["flops_used"]

    synthesized = _synthesize_sampling_breakdown(
        width=width,
        depth=depth,
        n_samples=n_samples,
        wall_time_s=0.0,  # wall time doesn't affect flops_used
    )

    assert synthesized["flops_used"] == actual_flops, (
        f"Closed-form FLOP count {synthesized['flops_used']} does not match "
        f"flopscope's actual count {actual_flops}. The synthesis formula in "
        f"src/whestbench/dataset_torch.py:_synthesize_sampling_breakdown "
        f"needs to be updated to match flopscope's current accounting model."
    )

    # The breakdown also needs the right shape for downstream consumers:
    assert "by_namespace" in synthesized
    assert "sampling" in synthesized["by_namespace"]
    assert "sample_layer_statistics" in synthesized["by_namespace"]["sampling"]["by_namespace"]
