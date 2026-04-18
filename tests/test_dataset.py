import whest as we

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
            we.testing.assert_array_equal(wa, wb)

        we.testing.assert_array_equal(bundle_a.all_layer_means[idx], bundle_b.all_layer_means[idx])
        we.testing.assert_array_equal(bundle_a.final_means[idx], bundle_b.final_means[idx])
    assert bundle_a.avg_variances == bundle_b.avg_variances


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
