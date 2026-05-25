import json

import flopscope.numpy as fnp
import numpy as np
import pytest

import whestbench.simulation as simulation
from whestbench import hardware as hardware_mod
from whestbench.dataset import create_dataset, iter_mlps, load_dataset, metadata


def test_create_and_load_roundtrip(tmp_path) -> None:
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=8,
        depth=2,
        seed=42,
        output_path=tmp_path / "test",
    )
    ds = load_dataset(out)
    assert len(ds) == 2
    mlps = list(iter_mlps(ds))
    assert len(mlps) == 2
    assert np.array(ds["all_layer_means"]).shape == (2, 2, 8)
    assert np.array(ds["final_means"]).shape == (2, 8)
    assert len(ds["avg_variance"]) == 2
    breakdowns = [json.loads(b) for b in ds["sampling_budget_breakdown"]]
    assert len(breakdowns) == 2
    assert breakdowns[0]["flops_used"] > 0
    assert "sampling.sample_layer_statistics" in breakdowns[0]["by_namespace"]
    # New assertions for backend tag + schema bump:
    md = metadata(ds)
    assert md["schema_version"] == "3.0"
    assert md["backend"] == "flopscope"
    assert "flop_budget" not in md
    for mlp in mlps:
        mlp.validate()
        assert mlp.width == 8
        assert mlp.depth == 2
        # 3.0 datasets carry a non-empty, slug-shaped name on every MLP.
        assert mlp.name and "-" in mlp.name


def test_create_dataset_is_reproducible_with_explicit_seed(tmp_path) -> None:
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        seed=1234,
        output_path=tmp_path / "seeded_a",
    )
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        seed=1234,
        output_path=tmp_path / "seeded_b",
    )

    ds_a = load_dataset(tmp_path / "seeded_a")
    ds_b = load_dataset(tmp_path / "seeded_b")

    assert len(ds_a) == len(ds_b) == 3
    for idx in range(len(ds_a)):
        mlp_a = list(iter_mlps(ds_a))[idx]
        mlp_b = list(iter_mlps(ds_b))[idx]
        for wa, wb in zip(mlp_a.weights, mlp_b.weights):
            fnp.testing.assert_array_equal(wa, wb)

        fnp.testing.assert_array_equal(ds_a[idx]["all_layer_means"], ds_b[idx]["all_layer_means"])
        fnp.testing.assert_array_equal(ds_a[idx]["final_means"], ds_b[idx]["final_means"])
    assert ds_a["avg_variance"] == ds_b["avg_variance"]
    breakdowns_a = [json.loads(b) for b in ds_a["sampling_budget_breakdown"]]
    breakdowns_b = [json.loads(b) for b in ds_b["sampling_budget_breakdown"]]
    for breakdown_a, breakdown_b in zip(breakdowns_a, breakdowns_b, strict=True):
        assert breakdown_a["flops_used"] == breakdown_b["flops_used"]
        assert sorted(breakdown_a["by_namespace"]) == sorted(breakdown_b["by_namespace"])
        for namespace in breakdown_a["by_namespace"]:
            assert (
                breakdown_a["by_namespace"][namespace]["flops_used"]
                == breakdown_b["by_namespace"][namespace]["flops_used"]
            )


def test_create_dataset_reports_sampling_chunk_progress(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, int | str]] = []
    monkeypatch.setattr(simulation, "_pick_chunk_size", lambda _width: 4)

    create_dataset(
        n_mlps=2,
        n_samples=10,
        width=4,
        depth=1,
        seed=42,
        output_path=tmp_path / "chunked_progress",
        progress=events.append,
    )

    generating = [event for event in events if event["phase"] == "generating"]
    sampling = [event for event in events if event["phase"] == "sampling"]
    assert generating == [
        {"phase": "generating", "completed": 1, "total": 2},
        {"phase": "generating", "completed": 2, "total": 2},
    ]
    assert sampling[0]["phase"] == "sampling"
    assert sampling[0]["completed"] == 1
    assert sampling[0]["total"] == 6
    assert sampling[0]["mlp_index"] == 1
    assert sampling[0]["mlp_name"] == "kathleen-munoz"
    assert sampling[0]["n_mlps"] == 2
    assert sampling[0]["unit"] == "chunks"
    assert sampling[-1]["phase"] == "sampling"
    assert sampling[-1]["completed"] == 6
    assert sampling[-1]["total"] == 6
    assert sampling[-1]["mlp_index"] == 2
    assert sampling[-1]["mlp_name"] == "sheri-nguyen"
    assert sampling[-1]["n_mlps"] == 2
    assert sampling[-1]["unit"] == "chunks"


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
        seed=7,
        output_path=tmp_path / "skip_fallbacks",
    )

    ds = load_dataset(out)
    md = metadata(ds)

    assert md["hardware"]["hostname"]
    assert md["hardware"]["cpu_count_logical"] > 0
    assert md["hardware"]["cpu_count_physical"] is None
    assert md["hardware"]["ram_total_bytes"] is None


def test_load_dataset_rejects_file_path(tmp_path) -> None:
    """load_dataset must raise InvalidDatasetError when given a file, not a directory."""
    from whestbench.dataset_io import InvalidDatasetError

    fake_file = tmp_path / "legacy.npz"
    fake_file.write_bytes(b"fake")

    with pytest.raises(InvalidDatasetError, match="file"):
        load_dataset(fake_file)


def test_load_dataset_rejects_wrong_schema_version(tmp_path) -> None:
    """load_dataset must raise InvalidDatasetError for wrong schema_version."""
    import json

    from whestbench.dataset_io import InvalidDatasetError

    dataset_dir = tmp_path / "wrong_schema"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "2.4",
                "seed_protocol": {
                    "name": "whestbench_seedsequence_hierarchy",
                    "version": "2.0",
                    "seeded": False,
                },
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "seed": 0,
                "n_mlps": 1,
                "n_samples": 1,
                "width": 4,
                "depth": 2,
                "hardware": {},
            }
        )
    )

    with pytest.raises(InvalidDatasetError, match="schema_version"):
        load_dataset(dataset_dir)


# ---------------------------------------------------------------------------
# 3.0 schema: per-MLP human-readable names
# ---------------------------------------------------------------------------


def test_create_dataset_persists_mlp_names_matching_seed_assignment(tmp_path) -> None:
    """`mlp.name` after load matches `assign_unique_names([m.seed for m in mlps])`."""
    from whestbench.naming import assign_unique_names

    out = create_dataset(
        n_mlps=3,
        n_samples=32,
        width=4,
        depth=2,
        seed=2024,
        output_path=tmp_path / "named",
    )
    ds = load_dataset(out)
    mlps = list(iter_mlps(ds))

    expected = assign_unique_names([m.seed for m in mlps])
    assert [m.name for m in mlps] == expected
    # And every name is slug-shaped.
    for m in mlps:
        assert m.name
        assert m.name.islower()
        assert "-" in m.name


def test_create_dataset_names_reproduce_at_same_seed(tmp_path) -> None:
    """Two bakes at the same `--seed` produce identical name lists."""
    out_a = create_dataset(
        n_mlps=4,
        n_samples=32,
        width=4,
        depth=2,
        seed=7777,
        output_path=tmp_path / "a",
    )
    out_b = create_dataset(
        n_mlps=4,
        n_samples=32,
        width=4,
        depth=2,
        seed=7777,
        output_path=tmp_path / "b",
    )
    names_a = [m.name for m in iter_mlps(load_dataset(out_a))]
    names_b = [m.name for m in iter_mlps(load_dataset(out_b))]
    assert names_a == names_b
    assert all(names_a)  # no empty strings
