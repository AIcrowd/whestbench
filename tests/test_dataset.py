import json

import flopscope.numpy as fnp
import numpy as np

from whestbench import hardware as hardware_mod
from whestbench.dataset import create_dataset, load_dataset


def test_create_and_load_roundtrip(tmp_path) -> None:
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=8,
        depth=2,
        flop_budget=32,
        seed=42,
        output_path=tmp_path / "test.npz",
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 2
    assert len(bundle.mlps) == 2
    assert bundle.all_layer_means.shape == (2, 2, 8)
    assert bundle.final_means.shape == (2, 8)
    assert len(bundle.avg_variances) == 2
    assert bundle.sampling_budget_breakdowns is not None
    assert len(bundle.sampling_budget_breakdowns) == 2
    assert bundle.sampling_budget_breakdowns[0]["flops_used"] > 0
    assert (
        "sampling.sample_layer_statistics" in bundle.sampling_budget_breakdowns[0]["by_namespace"]
    )
    for mlp in bundle.mlps:
        mlp.validate()
        assert mlp.width == 8
        assert mlp.depth == 2


def test_create_dataset_is_reproducible_with_explicit_seed(tmp_path) -> None:
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        flop_budget=32,
        seed=1234,
        output_path=tmp_path / "seeded_a.npz",
    )
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        flop_budget=32,
        seed=1234,
        output_path=tmp_path / "seeded_b.npz",
    )

    bundle_a = load_dataset(tmp_path / "seeded_a.npz")
    bundle_b = load_dataset(tmp_path / "seeded_b.npz")

    assert bundle_a.n_mlps == bundle_b.n_mlps == 3
    for idx in range(bundle_a.n_mlps):
        mlp_a = bundle_a.mlps[idx]
        mlp_b = bundle_b.mlps[idx]
        for wa, wb in zip(mlp_a.weights, mlp_b.weights):
            fnp.testing.assert_array_equal(wa, wb)

        fnp.testing.assert_array_equal(bundle_a.all_layer_means[idx], bundle_b.all_layer_means[idx])
        fnp.testing.assert_array_equal(bundle_a.final_means[idx], bundle_b.final_means[idx])
    assert bundle_a.avg_variances == bundle_b.avg_variances
    assert bundle_a.sampling_budget_breakdowns is not None
    assert bundle_b.sampling_budget_breakdowns is not None
    for breakdown_a, breakdown_b in zip(
        bundle_a.sampling_budget_breakdowns,
        bundle_b.sampling_budget_breakdowns,
        strict=True,
    ):
        assert breakdown_a["flops_used"] == breakdown_b["flops_used"]
        assert sorted(breakdown_a["by_namespace"]) == sorted(breakdown_b["by_namespace"])
        for namespace in breakdown_a["by_namespace"]:
            assert (
                breakdown_a["by_namespace"][namespace]["flops_used"]
                == breakdown_b["by_namespace"][namespace]["flops_used"]
            )


def test_create_dataset_skips_hardware_fallback_probes_via_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(hardware_mod._SKIP_FALLBACK_PROBES_ENV, "1")
    monkeypatch.setattr(hardware_mod, "psutil", None)

    def _unexpected_cpu_probe() -> int:
        raise AssertionError("physical core fallback probe should be skipped")

    def _unexpected_ram_probe() -> int:
        raise AssertionError("RAM fallback probe should be skipped")

    monkeypatch.setattr(hardware_mod, "_physical_core_count_fallback", _unexpected_cpu_probe)
    monkeypatch.setattr(hardware_mod, "_ram_total_fallback", _unexpected_ram_probe)

    out = create_dataset(
        n_mlps=1,
        n_samples=8,
        width=4,
        depth=2,
        flop_budget=32,
        seed=7,
        output_path=tmp_path / "skip_fallbacks.npz",
    )

    bundle = load_dataset(out)

    assert bundle.metadata["hardware"]["hostname"]
    assert bundle.metadata["hardware"]["cpu_count_logical"] > 0
    assert bundle.metadata["hardware"]["cpu_count_physical"] is None
    assert bundle.metadata["hardware"]["ram_total_bytes"] is None


def test_load_dataset_accepts_older_files_without_sampling_breakdowns(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((2, 2, 4, 4)).astype(np.float32)
    all_layer_means = rng.standard_normal((2, 2, 4)).astype(np.float32)
    final_means = rng.standard_normal((2, 4)).astype(np.float32)
    avg_variances = np.ones(2, dtype=np.float64)
    metadata = {
        "schema_version": "2.1",
        "created_at_utc": "2026-04-19T00:00:00+00:00",
        "seed": 0,
        "n_mlps": 2,
        "n_samples": 16,
        "width": 4,
        "depth": 2,
        "flop_budget": 1000,
        "hardware": {},
    }
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
    )

    bundle = load_dataset(path)

    assert bundle.n_mlps == 2
    assert bundle.sampling_budget_breakdowns is None
