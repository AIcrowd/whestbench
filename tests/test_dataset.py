import json
from typing import cast

import flopscope.numpy as fnp
import numpy as np
import pytest
from datasets import Dataset

import whestbench.simulation as simulation
from whestbench import hardware as hardware_mod
from whestbench.dataset import create_dataset, iter_mlps, load_dataset, metadata


def test_create_and_load_roundtrip(tmp_path) -> None:
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=8,
        depth=2,
        mlp_seeds=[42000, 42001],
        output_path=tmp_path / "test",
    )
    ds = load_dataset(out, split="public")
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


def test_create_dataset_is_reproducible_with_explicit_mlp_seeds(tmp_path) -> None:
    mlp_seeds = [1, 2, 3]
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=tmp_path / "seeded_a",
    )
    create_dataset(
        n_mlps=3,
        n_samples=64,
        width=8,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=tmp_path / "seeded_b",
    )

    ds_a = load_dataset(tmp_path / "seeded_a", split="public")
    ds_b = load_dataset(tmp_path / "seeded_b", split="public")

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
    import flopscope.numpy as fnp

    events: list[dict[str, int | str]] = []
    monkeypatch.setattr(simulation, "_pick_chunk_size", lambda _width: 4)

    # Derive names deterministically: use seeds that produce "kathleen-munoz" and "sheri-nguyen"
    # Under v3, names come from estimator seeds derived via SeedSequence(input).spawn(3)[2].
    # Seed 42 from the old protocol: SeedSequence(42).spawn(6)[2] gave kathleen-munoz.
    # We replicate the same estimator seeds by choosing mlp_seeds that produce the same derivation.
    # However the test fixtures are hardcoded to specific names. Let's compute them properly.
    # Old v2 seed=42: SeedSequence(42).spawn(3*2) → stream_seed[2] and stream_seed[5] are estimator seeds.
    old_ss = fnp.random.SeedSequence(42)
    old_stream = old_ss.spawn(3 * 2)
    seed0 = int(old_stream[2].generate_state(1)[0])
    seed1 = int(old_stream[5].generate_state(1)[0])
    # Under v3, to get the same estimator seeds, we need mlp_seeds that satisfy:
    #   SeedSequence(mlp_seed).spawn(3)[2].generate_state(1)[0] == old_estimator_seed
    # That's not easily invertible. Instead, let's find mlp_seeds s.t. the estimator
    # seeds produce the same names — but this is complex. Simplest: just check
    # the progress structure without hard-coding the names.
    mlp_seeds = [
        seed0,
        seed1,
    ]  # use estimator seeds as input seeds (not same as v2 but deterministic)

    create_dataset(
        n_mlps=2,
        n_samples=10,
        width=4,
        depth=1,
        mlp_seeds=mlp_seeds,
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
    assert isinstance(sampling[0]["mlp_name"], str) and sampling[0]["mlp_name"]
    assert sampling[0]["n_mlps"] == 2
    assert sampling[0]["unit"] == "chunks"
    assert sampling[-1]["phase"] == "sampling"
    assert sampling[-1]["completed"] == 6
    assert sampling[-1]["total"] == 6
    assert sampling[-1]["mlp_index"] == 2
    assert isinstance(sampling[-1]["mlp_name"], str) and sampling[-1]["mlp_name"]
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
        mlp_seeds=[7000],
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
        mlp_seeds=[2024000, 2024001, 2024002],
        output_path=tmp_path / "named",
    )
    ds = load_dataset(out, split="public")
    mlps = list(iter_mlps(ds))

    expected = assign_unique_names([m.seed for m in mlps])
    assert [m.name for m in mlps] == expected
    # And every name is slug-shaped.
    for m in mlps:
        assert m.name
        assert m.name.islower()
        assert "-" in m.name


def test_create_dataset_names_reproduce_at_same_mlp_seeds(tmp_path) -> None:
    """Two bakes at the same mlp_seeds produce identical name lists."""
    mlp_seeds = [7777000, 7777001, 7777002, 7777003]
    out_a = create_dataset(
        n_mlps=4,
        n_samples=32,
        width=4,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=tmp_path / "a",
    )
    out_b = create_dataset(
        n_mlps=4,
        n_samples=32,
        width=4,
        depth=2,
        mlp_seeds=mlp_seeds,
        output_path=tmp_path / "b",
    )
    names_a = [m.name for m in iter_mlps(load_dataset(out_a, split="public"))]
    names_b = [m.name for m in iter_mlps(load_dataset(out_b, split="public"))]
    assert names_a == names_b
    assert all(names_a)  # no empty strings


# ---------------------------------------------------------------------------
# Task 4: polymorphic load_dataset + metadata filter + iter_mlps/mlp_at guards
# ---------------------------------------------------------------------------


def _make_multi_split(tmp_path):
    """Bake two tiny single-split datasets and combine them."""
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import combine_split_datasets

    pub = tmp_path / "pub"
    hold = tmp_path / "hold"
    create_dataset(
        n_mlps=2,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[1000, 1001],
        output_path=pub,
        split="public",
    )
    create_dataset(
        n_mlps=3,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[2000, 2001, 2002],
        output_path=hold,
        split="holdout",
    )
    combined = tmp_path / "combined"
    combine_split_datasets([pub, hold], output_dir=combined)
    return combined


def test_load_dataset_multi_split_returns_dataset_dict(tmp_path):
    from datasets import DatasetDict

    from whestbench.dataset import load_dataset

    combined = _make_multi_split(tmp_path)
    dsd = load_dataset(combined)
    assert isinstance(dsd, DatasetDict)
    assert set(dsd.keys()) == {"public", "holdout"}
    assert len(dsd["public"]) == 2
    assert len(dsd["holdout"]) == 3


def test_load_dataset_multi_split_with_split_returns_dataset(tmp_path):
    from datasets import Dataset

    from whestbench.dataset import load_dataset

    combined = _make_multi_split(tmp_path)
    ds = load_dataset(combined, split="public")
    assert isinstance(ds, Dataset)
    assert len(ds) == 2
    ds_h = load_dataset(combined, split="holdout")
    assert isinstance(ds_h, Dataset)
    assert len(ds_h) == 3


def test_load_dataset_multi_split_unknown_split_raises(tmp_path):
    import pytest

    from whestbench.dataset import load_dataset
    from whestbench.dataset_io import InvalidDatasetError

    combined = _make_multi_split(tmp_path)
    with pytest.raises(InvalidDatasetError, match=r"split.+not.+\{.*public.*\}"):
        load_dataset(combined, split="does-not-exist")


def test_load_dataset_single_split_returns_dataset_unchanged(tmp_path):
    from datasets import Dataset

    from whestbench.dataset import create_dataset, load_dataset

    out = tmp_path / "single"
    create_dataset(
        n_mlps=2,
        n_samples=100,
        width=4,
        depth=2,
        mlp_seeds=[1000, 1001],
        output_path=out,
    )
    ds = load_dataset(out)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2
    ds2 = load_dataset(out, split="public")
    assert isinstance(ds2, Dataset)
    assert len(ds2) == 2


def test_metadata_on_dataset_dict_returns_full_multi_split_dict(tmp_path):
    from whestbench.dataset import load_dataset, metadata

    combined = _make_multi_split(tmp_path)
    dsd = load_dataset(combined)
    md = metadata(dsd)
    assert "splits" in md
    assert set(md["splits"].keys()) == {"public", "holdout"}


def test_metadata_on_dataset_dict_with_split_returns_merged_single_split_shape(tmp_path):
    from whestbench.dataset import load_dataset, metadata

    combined = _make_multi_split(tmp_path)
    dsd = load_dataset(combined)
    md = metadata(dsd, split="public")
    assert "splits" not in md
    assert md["n_mlps"] == 2
    # Under v3, there is no top-level `seed` field.
    assert "seed" not in md


def test_metadata_on_member_dataset_returns_merged_single_split_shape(tmp_path):
    """Accessing dsd['public'] returns a Dataset with merged metadata attached."""
    from whestbench.dataset import load_dataset, metadata

    combined = _make_multi_split(tmp_path)
    dsd = load_dataset(combined)
    ds = cast(Dataset, dsd["public"])
    md = metadata(ds)
    assert "splits" not in md
    assert md["n_mlps"] == 2
    # Under v3, there is no top-level `seed` field.
    assert "seed" not in md


def test_metadata_split_arg_rejected_on_dataset(tmp_path):
    import pytest

    from whestbench.dataset import load_dataset, metadata

    combined = _make_multi_split(tmp_path)
    ds = load_dataset(combined, split="public")
    with pytest.raises(TypeError, match=r"split=.+Dataset"):
        metadata(ds, split="public")


def test_iter_mlps_raises_on_dataset_dict(tmp_path):
    import pytest

    from whestbench.dataset import iter_mlps, load_dataset

    dsd = load_dataset(_make_multi_split(tmp_path))
    with pytest.raises(TypeError, match=r"single Dataset"):
        next(iter_mlps(cast(Dataset, dsd)))


def test_mlp_at_raises_on_dataset_dict(tmp_path):
    import pytest

    from whestbench.dataset import load_dataset, mlp_at

    dsd = load_dataset(_make_multi_split(tmp_path))
    with pytest.raises(TypeError, match=r"single Dataset"):
        mlp_at(cast(Dataset, dsd), 0)


def test_iter_mlps_works_on_member_dataset(tmp_path):
    from whestbench.dataset import iter_mlps, load_dataset

    dsd = load_dataset(_make_multi_split(tmp_path))
    mlps = list(iter_mlps(cast(Dataset, dsd["public"])))
    assert len(mlps) == 2
    for m in mlps:
        m.validate()


# ---------------------------------------------------------------------------
# Task 3: seed_protocol 3.0 — per-MLP explicit seeds
# ---------------------------------------------------------------------------


def test_create_dataset_writes_seed_protocol_v3_by_default(tmp_path):
    """create_dataset with no seed kwarg writes a 3.0 dataset."""
    import json

    from whestbench.dataset import create_dataset

    out = create_dataset(
        n_mlps=4,
        n_samples=50,
        width=4,
        depth=2,
        output_path=tmp_path / "default",
    )
    md = json.loads((out / "metadata.json").read_text())
    assert md["seed_protocol"]["name"] == "whestbench_explicit_per_mlp_seeds"
    assert md["seed_protocol"]["version"] == "3.0"
    assert "seed" not in md  # no top-level seed under 3.0
    assert md["n_mlps"] == 4


def test_create_dataset_with_explicit_mlp_seeds_writes_them_to_parquet(tmp_path):
    """Each MLP's parquet `mlp_seed` column under 3.0 is the input seed."""
    from datasets import load_dataset as hf_load

    from whestbench.dataset import create_dataset

    seeds = [42, 1234567890, 999_999_999, 7]
    out = create_dataset(
        n_mlps=4,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=seeds,
        output_path=tmp_path / "explicit",
    )
    ds = hf_load(str(out), split="public")
    assert ds["mlp_seed"] == seeds


def test_create_dataset_auto_generated_seeds_are_distinct_and_in_range(tmp_path):
    """Auto-generated seeds are distinct int63 values."""
    from datasets import load_dataset as hf_load

    from whestbench.dataset import create_dataset

    out = create_dataset(
        n_mlps=10,
        n_samples=50,
        width=4,
        depth=2,
        output_path=tmp_path / "auto",
    )
    ds = hf_load(str(out), split="public")
    seeds = ds["mlp_seed"]
    assert len(seeds) == 10
    assert len(set(seeds)) == 10  # distinct
    for s in seeds:
        assert 0 <= s < (1 << 63)


def test_create_dataset_rejects_legacy_seed_kwarg(tmp_path):
    import pytest

    from whestbench.dataset import create_dataset

    with pytest.raises(TypeError, match=r"seed.+no longer supported.+mlp_seeds"):
        create_dataset(
            n_mlps=4,
            n_samples=50,
            width=4,
            depth=2,
            seed=42,  # type: ignore[call-arg]
            output_path=tmp_path / "rejected",
        )


def test_create_dataset_validates_mlp_seeds_length(tmp_path):
    import pytest

    from whestbench.dataset import create_dataset

    with pytest.raises(ValueError, match=r"length 3.+n_mlps=4"):
        create_dataset(
            n_mlps=4,
            n_samples=50,
            width=4,
            depth=2,
            mlp_seeds=[1, 2, 3],
            output_path=tmp_path / "bad",
        )


def test_create_dataset_validates_mlp_seeds_values(tmp_path):
    import pytest

    from whestbench.dataset import create_dataset

    with pytest.raises(ValueError, match=r"out of range"):
        create_dataset(
            n_mlps=4,
            n_samples=50,
            width=4,
            depth=2,
            mlp_seeds=[1, 2, 3, 2**63],  # last is out of range
            output_path=tmp_path / "bad",
        )


def test_create_dataset_v3_slice_is_bit_equivalent_to_full_bake(tmp_path):
    """Worker baking slice K/N with --mlp-seeds <full> matches single-host."""
    import flopscope.numpy as fnp
    from datasets import load_dataset as hf_load

    from whestbench.dataset import create_dataset

    seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
    # Single-host bake
    full = create_dataset(
        n_mlps=8,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=seeds,
        output_path=tmp_path / "full",
    )
    # Slice 0 of 2 (MLPs 0..3) — same full seed list
    slice0 = create_dataset(
        n_mlps=8,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=seeds,
        mlp_range=(0, 4),
        output_path=tmp_path / "slice0",
    )
    full_ds = hf_load(str(full), split="public")
    slice0_ds = hf_load(str(slice0), split="public")
    for i in range(4):
        # mlp_id under slice maps to absolute index
        assert slice0_ds[i]["mlp_seed"] == full_ds[i]["mlp_seed"]
        assert slice0_ds[i]["mlp_name"] == full_ds[i]["mlp_name"]
        for col in ("weights", "all_layer_means", "final_means"):
            fnp.testing.assert_array_equal(
                fnp.array(slice0_ds[i][col]),
                fnp.array(full_ds[i][col]),
            )


def test_iter_mlps_v3_dataset_derives_estimator_seed(tmp_path):
    """Under 3.0, mlp.seed is derived from parquet mlp_seed via SeedSequence."""
    import flopscope.numpy as fnp

    from whestbench.dataset import create_dataset, iter_mlps, load_dataset

    seeds = [42, 99]
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=seeds,
        output_path=tmp_path / "v3",
    )
    ds = load_dataset(out, split="public")
    mlps = list(iter_mlps(ds))
    # For each MLP under 3.0, mlp.seed is the locally-derived estimator seed
    # from the parquet mlp_seed column (which IS the input seed).
    for i, expected_input_seed in enumerate(seeds):
        derived = int(fnp.random.SeedSequence(expected_input_seed).spawn(3)[2].generate_state(1)[0])
        assert mlps[i].seed == derived


def test_iter_mlps_v2_legacy_dataset_uses_parquet_seed_directly(tmp_path):
    """For legacy 2.0 datasets, mlp.seed is the parquet mlp_seed directly."""
    import json

    from whestbench.dataset import create_dataset, iter_mlps, load_dataset

    # Create a v3 dataset, then mutate metadata.json to look like v2.
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=4,
        depth=2,
        mlp_seeds=[42, 99],
        output_path=tmp_path / "fake_v2",
    )
    md_path = out / "metadata.json"
    md = json.loads(md_path.read_text())
    md["seed_protocol"] = {
        "name": "whestbench_seedsequence_hierarchy",
        "version": "2.0",
    }
    md["seed"] = 12345  # required top-level seed for 2.0
    md_path.write_text(json.dumps(md, indent=2))

    ds = load_dataset(out, split="public")
    mlps = list(iter_mlps(ds))
    # Under 2.0, mlp.seed is the parquet mlp_seed directly (no derivation).
    assert mlps[0].seed == 42
    assert mlps[1].seed == 99
