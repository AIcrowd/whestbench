import numpy as np
import pytest
from datasets import Dataset

from whestbench.scoring import ContestSpec, make_contest_from_dataset
from whestbench.seeds import derive_estimator_seed


def _ds(n=2, width=4, depth=2):
    rows = [
        {
            "mlp_id": i,
            "mlp_name": f"m-{i}",
            "mlp_seed": i * 1000 + 1,
            "weights": np.zeros((depth, width, width), dtype=np.float32).tolist(),
            "all_layer_means": np.zeros((depth, width), dtype=np.float32).tolist(),
            "final_means": np.zeros(width, dtype=np.float32).tolist(),
            "avg_variance": 0.1,
            "sampling_budget_breakdown": '{"flop_budget":0,"flops_used":0,"flops_remaining":0,"wall_time_s":0.0,"flopscope_backend_time_s":0.0,"flopscope_overhead_time_s":0.0,"residual_wall_time_s":0.0}',
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def _spec(n=2, width=4, depth=2):
    return ContestSpec(
        width=width, depth=depth, n_mlps=n, flop_budget=10**9, ground_truth_samples=8
    )


def test_unregistered_dataset_without_explicit_protocol_raises():
    with pytest.raises(ValueError, match="seed_protocol"):
        make_contest_from_dataset(_spec(), _ds(), n_mlps=2)


def test_explicit_protocol_3_0_uses_derived_seed():
    cd = make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="3.0")
    assert cd.mlps[0].seed == derive_estimator_seed(1)  # mlp_seed=1
    assert cd.mlps[1].seed == derive_estimator_seed(1001)


def test_explicit_protocol_2_0_uses_raw_seed():
    cd = make_contest_from_dataset(_spec(), _ds(), n_mlps=2, seed_protocol_version="2.0")
    assert cd.mlps[0].seed == 1
    assert cd.mlps[1].seed == 1001
