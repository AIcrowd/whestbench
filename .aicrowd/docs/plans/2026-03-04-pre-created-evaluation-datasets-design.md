# Pre-Created Evaluation Datasets

Pre-compute circuits, ground truth means, and sampling baselines so `cestim run` can skip the expensive generation/sampling phase on repeated evaluations.

## Commands

### `cestim create-dataset`

```
cestim create-dataset \
  --n-circuits 10 \
  --n-samples 10000 \
  --seed 42 \                          # optional, auto-generated if omitted
  --width 100 --max-depth 30 \         # default from contest params
  --budgets 10,100,1000,10000 \        # default from contest params
  -o eval_dataset.npz                  # default: ./eval_dataset.npz
```

Steps:
1. Generate `n_circuits` random circuits using the seed
2. Sample ground truth means via `empirical_mean()` for each circuit
3. Compute per-budget sampling baseline times
4. Collect hardware fingerprint
5. Pack everything into a single `.npz` file

### `cestim run --dataset`

```
cestim run --estimator my_estimator.py --dataset eval_dataset.npz [--strict-baselines]
```

- Loads circuits, ground truth, and baselines from the file
- On hardware mismatch: warns and auto-recomputes baselines
- `--strict-baselines`: refuses to run on hardware mismatch
- `--n-circuits` and `--n-samples` are ignored when `--dataset` is provided

## File Format

Single `.npz` archive with the following keys:

| Key | Type | Shape |
|-----|------|-------|
| `metadata` | JSON string (stored as numpy char array) | — |
| `circuits_first` | `int32` | `(n_circuits, depth, width)` |
| `circuits_second` | `int32` | `(n_circuits, depth, width)` |
| `circuits_coeff` | `float32` | `(n_circuits, depth, width, 4)` |
| `ground_truth_means` | `float32` | `(n_circuits, depth, width)` |
| `baseline_times` | `float64` | `(n_budgets, depth)` |

### Metadata JSON

```json
{
  "schema_version": "1.0",
  "created_at_utc": "2026-03-04T10:00:00+00:00",
  "seed": 42,
  "n_circuits": 10,
  "n_samples": 10000,
  "width": 100,
  "max_depth": 30,
  "budgets": [10, 100, 1000, 10000],
  "time_tolerance": 0.1,
  "hardware": { ... }
}
```

## Hardware Fingerprint

Shared `collect_hardware_fingerprint()` function used by both dataset creation and the CLI Hardware & Runtime dashboard pane.

| Field | Source |
|-------|--------|
| `hostname` | `socket.gethostname()` |
| `os` | `platform.system()` |
| `os_release` | `platform.release()` |
| `platform` | `platform.platform()` |
| `machine` | `platform.machine()` |
| `python_version` | `platform.python_version()` |
| `cpu_brand` | `platform.processor()` |
| `cpu_count_logical` | `os.cpu_count()` |
| `cpu_count_physical` | `psutil.cpu_count(logical=False)` (fallback: `None`) |
| `ram_total_bytes` | `psutil.virtual_memory().total` (fallback: `None`) |
| `ram_available_bytes` | `psutil.virtual_memory().available` (snapshot) |
| `numpy_version` | `np.__version__` |

### Staleness Check

Compares `machine` + `cpu_brand` + `cpu_count_logical` + `ram_total_bytes`. Mismatch triggers:
- Default: warning + auto-recompute baselines
- `--strict-baselines`: error, refuses to run

## Architecture

### New module: `dataset.py`

- `create_dataset(...)` — generates circuits, samples ground truth, computes baselines, saves `.npz`
- `load_dataset(path)` → `DatasetBundle` dataclass
- `check_hardware_staleness(stored, current)` → `bool`

### New module: `hardware.py`

- `collect_hardware_fingerprint()` → `dict` — shared by dataset, scoring, and CLI dashboard

### Modified: `cli.py`

- Add `create-dataset` subcommand
- Add `--dataset` and `--strict-baselines` flags to `run`

### Modified: `scoring.py`

- `score_estimator_report()` accepts optional pre-loaded circuits, ground truth means, and baseline times

### Modified: `reporting.py`

- `_hardware_runtime_panel()` uses enriched fingerprint (CPU cores, RAM, etc.)
