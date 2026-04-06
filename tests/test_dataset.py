from network_estimation.dataset import create_dataset, load_dataset


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
