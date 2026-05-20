"""Tests for the torch-backed dataset path. Skipped if torch is not installed."""

# pyright: reportMissingImports=false
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whestbench._simulation_torch import sample_layer_statistics_torch  # noqa: E402


def _seed_torch_generator(seed: int, device: str) -> Any:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def test_sample_layer_statistics_torch_returns_expected_shapes() -> None:
    width = 4
    depth = 3
    B = 2
    n_samples = 64

    weights_batch = torch.randn((B, depth, width, width), dtype=torch.float32)
    generators = [_seed_torch_generator(seed=i, device="cpu") for i in range(B)]

    layer_means, final_means, avg_variances = sample_layer_statistics_torch(
        weights_batch=weights_batch,
        n_samples=n_samples,
        generators=generators,
        chunk_size=16,
        progress=None,
    )

    assert layer_means.shape == (B, depth, width)
    assert layer_means.dtype == torch.float32
    assert final_means.shape == (B, width)
    assert final_means.dtype == torch.float32
    assert avg_variances.shape == (B,)
    assert avg_variances.dtype == torch.float64


def test_sample_layer_statistics_torch_emits_progress() -> None:
    weights_batch = torch.randn((1, 2, 4, 4), dtype=torch.float32)
    generators = [_seed_torch_generator(0, "cpu")]
    events: list[dict] = []

    sample_layer_statistics_torch(
        weights_batch=weights_batch,
        n_samples=20,
        generators=generators,
        chunk_size=10,
        progress=events.append,
    )

    assert len(events) == 2  # 20 samples / 10 chunk_size = 2 chunks
    assert events[0] == {"completed": 1, "total": 2, "unit": "chunks"}
    assert events[1] == {"completed": 2, "total": 2, "unit": "chunks"}


from whestbench.dataset_torch import _resolve_device  # noqa: E402


def test_resolve_device_explicit_cpu_returns_cpu() -> None:
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_auto_resolves_cpu_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto") == "mps"


def test_resolve_device_explicit_cuda_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but"):
        _resolve_device("cuda")


def test_resolve_device_explicit_mps_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS requested but"):
        _resolve_device("mps")


def test_resolve_device_invalid_raises() -> None:
    with pytest.raises(ValueError, match="device must be"):
        _resolve_device("tpu")


from whestbench.dataset_torch import _auto_chunk_size, _auto_mlps_per_batch  # noqa: E402


def test_auto_mlps_per_batch_clamps_to_16() -> None:
    assert _auto_mlps_per_batch(n_mlps=5) == 5
    assert _auto_mlps_per_batch(n_mlps=16) == 16
    assert _auto_mlps_per_batch(n_mlps=100) == 16


def test_auto_chunk_size_cpu_returns_default() -> None:
    assert _auto_chunk_size(device="cpu", width=256, mlps_per_batch=10) == 65536


def test_auto_chunk_size_mps_returns_default() -> None:
    assert _auto_chunk_size(device="mps", width=256, mlps_per_batch=10) == 65536


def test_auto_chunk_size_cuda_uses_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate ~8 GB free → 25% budget = 2GB cap kicks in
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    size = _auto_chunk_size(device="cuda", width=256, mlps_per_batch=10)
    # Clamp [65536, 1<<20] — for width=256, mlps=10, 2GB target: 2GB / (10*256*4) ≈ 209715
    # which falls inside [65536, 1048576], so we expect ~209K rounded
    assert 65536 <= size <= (1 << 20)


def test_auto_chunk_size_cuda_clamps_low(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tiny free memory → clamps to minimum 65536
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 1024**2, 16 * 1024**3))
    assert _auto_chunk_size(device="cuda", width=256, mlps_per_batch=10) == 65536


def test_auto_chunk_size_cuda_clamps_high(monkeypatch: pytest.MonkeyPatch) -> None:
    # Huge free memory → clamps to maximum 1<<20
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * 1024**3, 80 * 1024**3))
    assert _auto_chunk_size(device="cuda", width=4, mlps_per_batch=1) == (1 << 20)


from whestbench.dataset import load_dataset  # noqa: E402
from whestbench.dataset_torch import create_dataset_torch  # noqa: E402


def test_create_dataset_torch_roundtrip_cpu(tmp_path: Path) -> None:
    out = create_dataset_torch(
        n_mlps=2,
        n_samples=128,
        width=8,
        depth=2,
        seed=42,
        output_path=tmp_path / "torch_cpu.npz",
        device="cpu",
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 2
    assert bundle.all_layer_means.shape == (2, 2, 8)
    assert bundle.final_means.shape == (2, 8)
    assert len(bundle.avg_variances) == 2
    assert bundle.sampling_budget_breakdowns is not None
    assert len(bundle.sampling_budget_breakdowns) == 2
    assert bundle.sampling_budget_breakdowns[0]["flops_used"] > 0


def test_create_dataset_torch_metadata_includes_backend_info(tmp_path: Path) -> None:
    out = create_dataset_torch(
        n_mlps=1,
        n_samples=64,
        width=4,
        depth=2,
        seed=42,
        output_path=tmp_path / "torch_meta.npz",
        device="cpu",
    )
    bundle = load_dataset(out)
    assert bundle.metadata["schema_version"] == "2.3"
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["device"] == "cpu"
    assert "torch_version" in bundle.metadata
    assert "mlps_per_batch" in bundle.metadata
    assert "chunk_size" in bundle.metadata


def test_create_dataset_torch_is_deterministic_with_same_seed(tmp_path: Path) -> None:
    out_a = create_dataset_torch(
        n_mlps=2,
        n_samples=128,
        width=4,
        depth=2,
        seed=42,
        output_path=tmp_path / "torch_det_a.npz",
        device="cpu",
    )
    out_b = create_dataset_torch(
        n_mlps=2,
        n_samples=128,
        width=4,
        depth=2,
        seed=42,
        output_path=tmp_path / "torch_det_b.npz",
        device="cpu",
    )
    bundle_a = load_dataset(out_a)
    bundle_b = load_dataset(out_b)

    np.testing.assert_array_equal(bundle_a.all_layer_means, bundle_b.all_layer_means)
    np.testing.assert_array_equal(bundle_a.final_means, bundle_b.final_means)
    assert bundle_a.avg_variances == bundle_b.avg_variances


def test_create_dataset_torch_statistically_matches_cpu_path(tmp_path: Path) -> None:
    """Means produced by torch path agree with flopscope CPU path within MC noise."""
    from whestbench.dataset import create_dataset

    common_kwargs: dict[str, Any] = {
        "n_mlps": 2,
        "n_samples": 100_000,
        "width": 4,
        "depth": 2,
        "seed": 42,
    }
    out_cpu = create_dataset(**common_kwargs, output_path=tmp_path / "cpu.npz")
    out_torch = create_dataset_torch(
        **common_kwargs, output_path=tmp_path / "torch.npz", device="cpu"
    )

    bundle_cpu = load_dataset(out_cpu)
    bundle_torch = load_dataset(out_torch)

    # Shapes must agree (weights are generated identically via numpy RNG):
    assert bundle_cpu.all_layer_means.shape == bundle_torch.all_layer_means.shape
    # final_means agree within 5σ of MC noise — σ ≈ 1/sqrt(N), tol = 5σ.
    # If this ever flakes, do NOT silently widen the tolerance — investigate
    # whether the torch hot loop is producing biased output or the seed
    # protocol has drifted. The 5σ bound at N=100K is generous enough to
    # absorb legitimate RNG variance between numpy PCG64 and torch Philox.
    tol = 5.0 / (common_kwargs["n_samples"] ** 0.5)
    np.testing.assert_allclose(
        bundle_cpu.final_means,
        bundle_torch.final_means,
        atol=tol,
        err_msg="Torch final_means diverge from CPU beyond MC noise tolerance",
    )


def test_mini_batch_correctness_uneven_n_mlps(tmp_path: Path) -> None:
    """n_mlps=7 with mlps_per_batch=3 → 3 batches (3+3+1); all MLPs processed."""
    out = create_dataset_torch(
        n_mlps=7,
        n_samples=64,
        width=4,
        depth=2,
        seed=42,
        output_path=tmp_path / "uneven.npz",
        device="cpu",
        mlps_per_batch=3,
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 7
    assert bundle.all_layer_means.shape == (7, 2, 4)
    assert bundle.final_means.shape == (7, 4)
    assert len(bundle.avg_variances) == 7
    assert bundle.sampling_budget_breakdowns is not None
    assert len(bundle.sampling_budget_breakdowns) == 7

    # Each MLP must have a distinct seed (from the SeedSequence protocol):
    seeds = [m.seed for m in bundle.mlps]
    assert len(set(seeds)) == 7


def test_mini_batch_equivalence_to_single_batch(tmp_path: Path) -> None:
    """Splitting into mini-batches must produce the same output as one big batch."""
    common: dict[str, Any] = {
        "n_mlps": 4,
        "n_samples": 128,
        "width": 4,
        "depth": 2,
        "seed": 42,
        "device": "cpu",
    }
    out_big = create_dataset_torch(**common, output_path=tmp_path / "big.npz", mlps_per_batch=4)
    out_small = create_dataset_torch(**common, output_path=tmp_path / "small.npz", mlps_per_batch=1)

    bundle_big = load_dataset(out_big)
    bundle_small = load_dataset(out_small)

    np.testing.assert_array_equal(bundle_big.all_layer_means, bundle_small.all_layer_means)
    np.testing.assert_array_equal(bundle_big.final_means, bundle_small.final_means)
    assert bundle_big.avg_variances == bundle_small.avg_variances


def test_progress_events_have_expected_schema(tmp_path: Path) -> None:
    events: list[dict] = []
    create_dataset_torch(
        n_mlps=2,
        n_samples=128,
        width=4,
        depth=2,
        seed=42,
        output_path=tmp_path / "progress.npz",
        device="cpu",
        progress=events.append,
        chunk_size=64,
    )

    generating = [e for e in events if e.get("phase") == "generating"]
    sampling = [e for e in events if e.get("phase") == "sampling"]

    # Generating phase: one event per MLP
    assert len(generating) == 2
    assert generating[-1] == {"phase": "generating", "completed": 2, "total": 2}

    # Sampling phase: each event has the required keys
    assert len(sampling) >= 1
    last = sampling[-1]
    assert last["phase"] == "sampling"
    assert last["unit"] == "chunks"
    assert last["completed"] == last["total"]  # final event reports completion
    assert "mlp_index_range" in last
    assert last["n_mlps"] == 2


require_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
require_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)


@require_cuda
def test_cuda_smoke_roundtrip(tmp_path: Path) -> None:
    out = create_dataset_torch(
        n_mlps=2,
        n_samples=256,
        width=8,
        depth=2,
        seed=42,
        output_path=tmp_path / "cuda_smoke.npz",
        device="cuda",
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 2
    assert bundle.metadata["device"] == "cuda"
    assert "cuda_device_name" in bundle.metadata
    assert "cuda_device_capability" in bundle.metadata


@require_mps
def test_mps_smoke_roundtrip(tmp_path: Path) -> None:
    out = create_dataset_torch(
        n_mlps=2,
        n_samples=256,
        width=8,
        depth=2,
        seed=42,
        output_path=tmp_path / "mps_smoke.npz",
        device="mps",
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 2
    assert bundle.metadata["device"] == "mps"
    assert "mps_device_name" in bundle.metadata
    # Sanity check: avg_variances should be non-negative
    assert all(v >= 0 for v in bundle.avg_variances)
    # FLOP count should be positive
    assert bundle.sampling_budget_breakdowns is not None
    assert bundle.sampling_budget_breakdowns[0]["flops_used"] > 0
