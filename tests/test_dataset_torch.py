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
