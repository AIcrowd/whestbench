from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import circuit as circuit_module


@pytest.fixture(autouse=True)
def _disable_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(circuit_module, "tqdm", lambda iterable: iterable)
