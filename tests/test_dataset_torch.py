"""Tests for the torch-backed dataset path. Skipped if torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from whestbench._simulation_torch import sample_layer_statistics_torch  # noqa: E402


def _seed_torch_generator(seed: int, device: str) -> "torch.Generator":
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
