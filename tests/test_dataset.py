import whest as we

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
