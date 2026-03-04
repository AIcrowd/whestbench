# Pre-Created Evaluation Datasets Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Add `cestim create-dataset` command to pre-compute circuits, ground truth, and baselines into a `.npz` file, and extend `cestim run --dataset` to consume it.

**Architecture:** New `hardware.py` for shared fingerprinting, new `dataset.py` for dataset I/O, modifications to `cli.py` (new subcommand + run flags) and `scoring.py` (accept preloaded data). Dashboard updated for richer hardware info.

**Tech Stack:** numpy `.npz`, `psutil` (optional), existing `platform`/`socket`/`os` stdlib modules.

---

### Task 1: Shared Hardware Fingerprint — `hardware.py`

**Files:**
- Create: `src/circuit_estimation/hardware.py`
- Test: `tests/test_hardware.py`

**Step 1: Write the failing test**

```python
# tests/test_hardware.py
from __future__ import annotations

from circuit_estimation.hardware import collect_hardware_fingerprint, hardware_matches


def test_collect_hardware_fingerprint_returns_required_keys():
    fp = collect_hardware_fingerprint()
    required = {
        "hostname", "os", "os_release", "platform", "machine",
        "python_version", "cpu_brand", "cpu_count_logical",
        "cpu_count_physical", "ram_total_bytes", "ram_available_bytes",
        "numpy_version",
    }
    assert required <= set(fp.keys())
    assert isinstance(fp["cpu_count_logical"], int)
    assert fp["cpu_count_logical"] > 0


def test_hardware_matches_same_machine():
    fp = collect_hardware_fingerprint()
    assert hardware_matches(fp, fp) is True


def test_hardware_matches_detects_mismatch():
    fp = collect_hardware_fingerprint()
    altered = {**fp, "cpu_brand": "FakeProcessor"}
    assert hardware_matches(fp, altered) is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/hardware.py
"""Shared hardware fingerprinting for dataset staleness and CLI reporting."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

import numpy as np

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    psutil = None


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Collect a hardware fingerprint dict for the current machine."""
    fp: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_brand": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or 1,
        "cpu_count_physical": None,
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "numpy_version": np.__version__,
    }
    if psutil is not None:
        try:
            fp["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except Exception:
            pass
        try:
            mem = psutil.virtual_memory()
            fp["ram_total_bytes"] = int(mem.total)
            fp["ram_available_bytes"] = int(mem.available)
        except Exception:
            pass
    return fp


_STALENESS_KEYS = ("machine", "cpu_brand", "cpu_count_logical", "ram_total_bytes")


def hardware_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True if the stored and current fingerprints match on key fields."""
    return all(stored.get(k) == current.get(k) for k in _STALENESS_KEYS)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/circuit_estimation/hardware.py tests/test_hardware.py
git commit -m "feat: add shared hardware fingerprint module"
```

---

### Task 2: Wire Existing Code to Use `hardware.py`

**Files:**
- Modify: `src/circuit_estimation/cli.py` — replace `_host_metadata()` with `collect_hardware_fingerprint()`
- Modify: `src/circuit_estimation/scoring.py` — replace inline `host_meta` dict with `collect_hardware_fingerprint()`
- Modify: `src/circuit_estimation/reporting.py` — extend `_hardware_runtime_panel()` to show new fields
- Modify: `src/circuit_estimation/__init__.py` — export `collect_hardware_fingerprint`

**Step 1: Replace `_host_metadata()` in `cli.py`**

Replace the `_host_metadata()` function and its usage at line 180 with:
```python
from .hardware import collect_hardware_fingerprint
```
And use `collect_hardware_fingerprint()` wherever `_host_metadata()` was called.

**Step 2: Replace inline `host_meta` in `scoring.py`**

Replace the inline dict at lines 427-434 with:
```python
from .hardware import collect_hardware_fingerprint
# ...
host_meta = collect_hardware_fingerprint()
```

**Step 3: Extend `_hardware_runtime_panel()` in `reporting.py`**

Add rows for CPU Brand, CPU Cores (logical), CPU Cores (physical), RAM Total, and NumPy Version.

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py src/circuit_estimation/scoring.py src/circuit_estimation/reporting.py src/circuit_estimation/__init__.py
git commit -m "refactor: wire cli/scoring/reporting to shared hardware fingerprint"
```

---

### Task 3: Dataset I/O Module — `dataset.py`

**Files:**
- Create: `src/circuit_estimation/dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from circuit_estimation.dataset import create_dataset, load_dataset, DatasetBundle


def test_create_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "ds.npz"
    create_dataset(
        n_circuits=2,
        n_samples=100,
        width=4,
        max_depth=2,
        budgets=[10, 100],
        seed=42,
        output_path=out,
    )
    assert out.exists()

    bundle = load_dataset(out)
    assert bundle.n_circuits == 2
    assert bundle.metadata["seed"] == 42
    assert bundle.metadata["n_samples"] == 100
    assert bundle.ground_truth_means.shape == (2, 2, 4)
    assert bundle.baseline_times.shape == (2, 2)  # n_budgets x depth
    assert len(bundle.circuits) == 2
    assert bundle.circuits[0].n == 4
    assert bundle.circuits[0].d == 2


def test_create_dataset_auto_seed(tmp_path: Path):
    out = tmp_path / "ds.npz"
    create_dataset(
        n_circuits=1, n_samples=50, width=4, max_depth=1,
        budgets=[10], output_path=out,
    )
    bundle = load_dataset(out)
    assert "seed" in bundle.metadata
    assert isinstance(bundle.metadata["seed"], int)


def test_create_dataset_reproducible(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    for path in (a, b):
        create_dataset(
            n_circuits=2, n_samples=100, width=4, max_depth=2,
            budgets=[10], seed=123, output_path=path,
        )
    ba = load_dataset(a)
    bb = load_dataset(b)
    np.testing.assert_array_equal(ba.ground_truth_means, bb.ground_truth_means)


def test_load_dataset_validates_schema(tmp_path: Path):
    bad_file = tmp_path / "bad.npz"
    np.savez(bad_file, metadata=np.array("{}"))
    with pytest.raises(ValueError, match="schema_version"):
        load_dataset(bad_file)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

```python
# src/circuit_estimation/dataset.py
"""Create, save, and load pre-computed evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit, Layer
from .generation import random_circuit
from .hardware import collect_hardware_fingerprint
from .scoring import sampling_baseline_time
from .simulation import empirical_mean

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """In-memory representation of a loaded evaluation dataset."""
    metadata: dict[str, Any]
    circuits: list[Circuit]
    ground_truth_means: NDArray[np.float32]
    baseline_times: NDArray[np.float64]

    @property
    def n_circuits(self) -> int:
        return len(self.circuits)


def create_dataset(
    *,
    n_circuits: int,
    n_samples: int,
    width: int,
    max_depth: int,
    budgets: list[int],
    seed: int | None = None,
    time_tolerance: float = 0.1,
    output_path: Path | str,
    progress: Any | None = None,
) -> Path:
    """Generate circuits, sample ground truth, compute baselines, and save."""
    output_path = Path(output_path)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)

    # Generate circuits
    circuits = [random_circuit(width, max_depth, rng) for _ in range(n_circuits)]

    # Pack circuit arrays
    circuits_first = np.stack(
        [np.stack([layer.first for layer in c.gates]) for c in circuits]
    ).astype(np.int32)
    circuits_second = np.stack(
        [np.stack([layer.second for layer in c.gates]) for c in circuits]
    ).astype(np.int32)
    circuits_coeff = np.stack([
        np.stack([
            np.stack([layer.const, layer.first_coeff, layer.second_coeff, layer.product_coeff], axis=-1)
            for layer in c.gates
        ])
        for c in circuits
    ]).astype(np.float32)

    # Sample ground truth
    means_list: list[list[NDArray[np.float32]]] = []
    for i, circuit in enumerate(circuits):
        means_list.append(list(empirical_mean(circuit, n_samples)))
        if progress is not None:
            progress({"phase": "sampling", "completed": i + 1, "total": n_circuits})
    ground_truth_means = np.array(means_list, dtype=np.float32)

    # Compute baselines
    baseline_rows: list[list[float]] = []
    for budget in budgets:
        baseline_rows.append(sampling_baseline_time(budget, width, max_depth))
    baseline_times = np.array(baseline_rows, dtype=np.float64)

    # Metadata
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_circuits": n_circuits,
        "n_samples": n_samples,
        "width": width,
        "max_depth": max_depth,
        "budgets": budgets,
        "time_tolerance": time_tolerance,
        "hardware": collect_hardware_fingerprint(),
    }

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        circuits_first=circuits_first,
        circuits_second=circuits_second,
        circuits_coeff=circuits_coeff,
        ground_truth_means=ground_truth_means,
        baseline_times=baseline_times,
    )
    return output_path


def load_dataset(path: Path | str) -> DatasetBundle:
    """Load a dataset bundle from a .npz file."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    metadata_raw = str(data["metadata"])
    metadata = json.loads(metadata_raw)

    if "schema_version" not in metadata:
        raise ValueError(
            "Invalid dataset file: missing 'schema_version' in metadata."
        )

    circuits_first = data["circuits_first"]
    circuits_second = data["circuits_second"]
    circuits_coeff = data["circuits_coeff"]
    ground_truth_means = data["ground_truth_means"].astype(np.float32)
    baseline_times = data["baseline_times"].astype(np.float64)

    n_circuits = int(circuits_first.shape[0])
    depth = int(circuits_first.shape[1])
    width = int(circuits_first.shape[2])

    circuits: list[Circuit] = []
    for i in range(n_circuits):
        gates: list[Layer] = []
        for j in range(depth):
            coeff = circuits_coeff[i, j]  # shape (width, 4)
            gates.append(Layer(
                first=circuits_first[i, j].astype(np.int32),
                second=circuits_second[i, j].astype(np.int32),
                const=coeff[:, 0].astype(np.float32),
                first_coeff=coeff[:, 1].astype(np.float32),
                second_coeff=coeff[:, 2].astype(np.float32),
                product_coeff=coeff[:, 3].astype(np.float32),
            ))
        circuit = Circuit(n=width, d=depth, gates=gates)
        circuit.validate()
        circuits.append(circuit)

    return DatasetBundle(
        metadata=metadata,
        circuits=circuits,
        ground_truth_means=ground_truth_means,
        baseline_times=baseline_times,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/circuit_estimation/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset create/load module"
```

---

### Task 4: Wire `score_estimator_report()` to Accept Preloaded Data

**Files:**
- Modify: `src/circuit_estimation/scoring.py` — add `ground_truth_means` and `baseline_times_by_budget` optional params
- Test: `tests/test_scoring_module.py` — add test for preloaded data path

**Step 1: Write the failing test**

```python
# Append to tests/test_scoring_module.py
def test_score_estimator_report_with_preloaded_data():
    """score_estimator_report uses preloaded means and baselines when provided."""
    width, depth = 4, 2
    circuits = [_constant_circuit(width, depth)]
    # Pre-computed ground truth means (matching constant circuit output)
    means = np.array([list(empirical_mean(circuits[0], 100))], dtype=np.float32)
    # Fake baseline times
    baselines = {10: np.ones(depth, dtype=np.float64) * 0.01}
    params = ContestParams(width=width, max_depth=depth, budgets=[10], time_tolerance=0.1)

    def estimator(_circuit: Circuit, _budget: int):
        for _ in range(depth):
            yield np.zeros(width, dtype=np.float32)

    report = score_estimator_report(
        estimator,
        n_circuits=1,
        n_samples=100,  # ignored when preloaded
        contest_params=params,
        circuits=circuits,
        ground_truth_means=means,
        baseline_times_by_budget=baselines,
    )
    assert "final_score" in report["results"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scoring_module.py::test_score_estimator_report_with_preloaded_data -v`
Expected: FAIL (unexpected keyword argument)

**Step 3: Modify `score_estimator_report()` signature**

Add two optional parameters:
- `ground_truth_means: NDArray[np.float32] | None = None`
- `baseline_times_by_budget: dict[int, NDArray[np.float64]] | None = None`

When `ground_truth_means` is provided, skip the `empirical_mean()` loop.
When `baseline_times_by_budget` is provided, use `baseline_times_by_budget[budget]` instead of calling `sampling_baseline_time()`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_scoring_module.py -v`
Expected: All PASS (existing + new)

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "feat: allow score_estimator_report to accept preloaded ground truth and baselines"
```

---

### Task 5: CLI — `create-dataset` Subcommand

**Files:**
- Modify: `src/circuit_estimation/cli.py` — add `create-dataset` parser + handler
- Test: `tests/test_cli_participant_commands.py` — add CLI test

**Step 1: Write the failing test**

```python
# Append to tests/test_cli_participant_commands.py
def test_create_dataset_command_produces_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "ds.npz"
    # Use tiny params to keep test fast
    exit_code = cli.main([
        "create-dataset",
        "--n-circuits", "1",
        "--n-samples", "50",
        "--width", "4",
        "--max-depth", "1",
        "--budgets", "10",
        "--seed", "42",
        "-o", str(out),
    ])
    assert exit_code == 0
    assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_participant_commands.py::test_create_dataset_command_produces_npz -v`
Expected: FAIL

**Step 3: Add the subparser and handler in `cli.py`**

In `_build_participant_parser()` add:
```python
create_ds_parser = subparsers.add_parser("create-dataset", help="Pre-create evaluation dataset.")
create_ds_parser.add_argument("--n-circuits", type=int, default=10)
create_ds_parser.add_argument("--n-samples", type=int, default=10000)
create_ds_parser.add_argument("--width", type=int, default=None)
create_ds_parser.add_argument("--max-depth", type=int, default=None)
create_ds_parser.add_argument("--budgets", type=str, default=None)
create_ds_parser.add_argument("--seed", type=int, default=None)
create_ds_parser.add_argument("-o", "--output", default="eval_dataset.npz")
create_ds_parser.add_argument("--json", dest="json_output", action="store_true")
create_ds_parser.add_argument("--debug", action="store_true")
```

In `_main_participant()` handle `command == "create-dataset"`:
```python
if command == "create-dataset":
    from .dataset import create_dataset
    contest = _default_contest_params()
    width = args.width or contest.width
    max_depth = args.max_depth or contest.max_depth
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else contest.budgets
    out = create_dataset(
        n_circuits=args.n_circuits,
        n_samples=args.n_samples,
        width=width,
        max_depth=max_depth,
        budgets=budgets,
        seed=args.seed,
        output_path=Path(args.output),
    )
    payload = {"ok": True, "path": str(out)}
    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Dataset created: {out}")
    return 0
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cli_participant_commands.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli_participant_commands.py
git commit -m "feat(cli): add create-dataset subcommand"
```

---

### Task 6: CLI — `run --dataset` and `--strict-baselines` Flags

**Files:**
- Modify: `src/circuit_estimation/cli.py` — add flags to `run` subcommand, wire dataset loading
- Modify: `src/circuit_estimation/dataset.py` — add `dataset_file_hash()` utility
- Test: `tests/test_cli_participant_commands.py` — add test

**Step 1: Add `dataset_file_hash()` to `dataset.py`**

```python
import hashlib

def dataset_file_hash(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a dataset file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

**Step 2: Write the failing test**

```python
# Append to tests/test_cli_participant_commands.py
def test_run_with_dataset_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    # First create a dataset
    ds_path = tmp_path / "ds.npz"
    cli.main([
        "create-dataset",
        "--n-circuits", "1", "--n-samples", "50",
        "--width", "4", "--max-depth", "1",
        "--budgets", "10", "--seed", "1",
        "-o", str(ds_path),
    ])
    # Write a trivial estimator
    est = tmp_path / "est.py"
    est.write_text(
        "import numpy as np\n"
        "from circuit_estimation.protocol import Estimator\n"
        "class MyEstimator(Estimator):\n"
        "    def setup(self, context): pass\n"
        "    def predict(self, circuit, budget):\n"
        "        for _ in range(circuit.d):\n"
        "            yield np.zeros(circuit.n, dtype=np.float32)\n"
        "    def teardown(self): pass\n"
    )
    exit_code = cli.main([
        "run", "--estimator", str(est),
        "--dataset", str(ds_path),
        "--runner", "inprocess",
        "--json",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "final_score" in report["results"]
    # Verify dataset reference is included in results
    ds_info = report["run_config"]["dataset"]
    assert ds_info["sha256"]  # non-empty hash
    assert ds_info["seed"] == 1
    assert ds_info["n_circuits"] == 1
    assert ds_info["n_samples"] == 50
    assert isinstance(ds_info["baselines_recomputed"], bool)
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_participant_commands.py::test_run_with_dataset_flag -v`
Expected: FAIL (unrecognized arguments: --dataset)

**Step 4: Add flags and wire the dataset path in `cli.py`**

Add to `run_parser`:
```python
run_parser.add_argument("--dataset", default=None, help="Path to pre-created dataset .npz file.")
run_parser.add_argument("--strict-baselines", action="store_true", help="Refuse to run if dataset hardware differs.")
```

In the `run` handler, when `args.dataset` is set:
1. Load dataset via `load_dataset()`
2. Compute SHA-256 hash via `dataset_file_hash()`
3. Compare hardware via `hardware_matches()` and `collect_hardware_fingerprint()`
4. If mismatch and `--strict-baselines`: error
5. If mismatch: warn + recompute baselines (set `baselines_recomputed = True`)
6. Pass circuits, ground_truth_means, and baselines to `score_estimator_report()`
7. Inject `dataset` reference into `report["run_config"]`:

```python
report["run_config"]["dataset"] = {
    "path": str(Path(args.dataset).resolve()),
    "sha256": dataset_file_hash(args.dataset),
    "seed": bundle.metadata.get("seed"),
    "n_circuits": bundle.metadata.get("n_circuits"),
    "n_samples": bundle.metadata.get("n_samples"),
    "baselines_recomputed": baselines_recomputed,
}
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_cli_participant_commands.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/circuit_estimation/cli.py src/circuit_estimation/dataset.py tests/test_cli_participant_commands.py
git commit -m "feat(cli): add --dataset and --strict-baselines flags to run command"
```

---

### Task 7: Update Dashboard & Final Integration

**Files:**
- Modify: `src/circuit_estimation/reporting.py` — enrich hardware panel
- Test: `tests/test_reporting.py` — verify new fields render

**Step 1: Update `_hardware_runtime_panel()`**

Add rows for:
- `CPU Brand [host.cpu_brand]`
- `CPU Cores (logical) [host.cpu_count_logical]`
- `CPU Cores (physical) [host.cpu_count_physical]`
- `RAM Total [host.ram_total_bytes]` (formatted as GB)
- `NumPy [host.numpy_version]`

**Step 2: Run existing reporting tests**

Run: `uv run pytest tests/test_reporting.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/circuit_estimation/reporting.py
git commit -m "feat(dashboard): enrich Hardware & Runtime panel with CPU/RAM/NumPy info"
```

---

### Task 8: User-Facing Documentation

**Files:**
- Create: `docs/how-to/use-evaluation-datasets.md`
- Modify: `docs/reference/cli-reference.md` — add `create-dataset` command and `run --dataset` flags
- Modify: `docs/index.md` — add link under "📈 I want to improve score" and Full Taxonomy

**Step 1: Write the how-to guide**

Create `docs/how-to/use-evaluation-datasets.md` with user-focused content:

```markdown
# Use Evaluation Datasets

## When to use this

Use pre-created evaluation datasets when you want to:

- **Skip repeated sampling** — circuit generation and ground truth sampling happen once, not on every run
- **Compare estimators fairly** — all estimators are scored against the exact same circuits
- **Iterate faster** — `cestim run --dataset` skips the "Sampling (Ground Truth)" phase entirely

## Create a dataset

```bash
cestim create-dataset -o my_dataset.npz
```

This generates circuits, samples ground truth means, and computes sampling baselines. All parameters are stored in the file so results are fully reproducible.

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-circuits` | 10 | Number of circuits to generate |
| `--n-samples` | 10000 | Samples per circuit for ground truth |
| `--seed` | auto | RNG seed (auto-generated if omitted, always recorded) |
| `-o, --output` | `eval_dataset.npz` | Output file path |
| `--width` | 100 | Wire count per circuit |
| `--max-depth` | 30 | Layers per circuit |
| `--budgets` | `10,100,1000,10000` | Comma-separated budget list |

## Run with a dataset

```bash
cestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

The `--n-circuits` and `--n-samples` flags are ignored when `--dataset` is provided — these values come from the dataset file.

## Hardware and baseline handling

The dataset records which machine it was created on. When you run on a different machine:

- **Default**: baselines are automatically recomputed (a warning is printed)
- **`--strict-baselines`**: refuses to run if hardware differs

```bash
# Strict mode — error if hardware doesn't match
cestim run --estimator ./estimator.py --dataset my_dataset.npz --strict-baselines
```

## Dataset traceability

When using `--dataset`, the results JSON includes a `dataset` reference under `run_config`:

```json
{
  "run_config": {
    "dataset": {
      "path": "/path/to/my_dataset.npz",
      "sha256": "a1b2c3...",
      "seed": 42,
      "n_circuits": 10,
      "n_samples": 10000,
      "baselines_recomputed": false
    }
  }
}
```

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)
```

**Step 2: Update CLI reference**

In `docs/reference/cli-reference.md`:

1. Add `cestim create-dataset` to the entry commands list
2. Add a `## cestim create-dataset` section with synopsis and key options
3. Add `--dataset` and `--strict-baselines` to the `cestim run` key options list

**Step 3: Update docs index**

In `docs/index.md`:

1. Under "📈 I want to improve score", add: `[Use Evaluation Datasets](./how-to/use-evaluation-datasets.md)`
2. In the Full Taxonomy How-To line, add the new guide

**Step 4: Commit**

```bash
git add docs/how-to/use-evaluation-datasets.md docs/reference/cli-reference.md docs/index.md
git commit -m "docs: add how-to guide for evaluation datasets and update CLI reference"
```

---

### Task 9: Manual End-to-End Verification

Run the full user workflow:

```bash
# Create a dataset
cestim create-dataset --n-circuits 3 --n-samples 1000 --seed 42 -o /tmp/test_ds.npz

# Run with the dataset
cestim run --estimator examples/estimators/mean_propagation.py --dataset /tmp/test_ds.npz

# Run with --json to verify report structure
cestim run --estimator examples/estimators/mean_propagation.py --dataset /tmp/test_ds.npz --json

# Run again to confirm it's faster (no sampling phase)
cestim run --estimator examples/estimators/combined_estimator.py --dataset /tmp/test_ds.npz
```

Expected:
- `create-dataset` produces `.npz` file with reported path
- `run --dataset` skips "Sampling (Ground Truth)" phase
- Report includes correct score and `run_config.dataset` reference
- Second run is noticeably faster than a fresh `cestim run` without `--dataset`

