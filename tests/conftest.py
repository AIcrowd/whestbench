import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import circuit as circuit_module


@pytest.fixture(autouse=True)
def _disable_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(circuit_module, "tqdm", lambda iterable: iterable)
