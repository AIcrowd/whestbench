# Profiler CLI Output Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the `profile-simulation` CLI output from a wall of tables into a compact leaderboard + detail view, with `--verbose` for the full dump.

**Architecture:** Add two new formatting functions (`format_compact_output` and `format_verbose_output`) to `profiler.py`. The existing `format_terminal_table` becomes `format_verbose_output`. A new `format_compact_output` renders the 3-zone layout. The `run_profile()` function gains a `verbose: bool` parameter. CLI gains `--verbose` and `--backends-help` flags.

**Tech Stack:** Python 3.10+, Rich (for tables/styling), shutil (for terminal width detection)

**Spec:** `docs/superpowers/specs/2026-03-20-profiler-cli-output-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/network_estimation/profiler.py` | Modify (lines 404-535, 582-765) | Add `format_compact_output()`, rename `format_terminal_table()` → `format_verbose_output()`, add `verbose` param to `run_profile()` |
| `src/network_estimation/cli.py` | Modify (lines 502-529, 886-899) | Add `--verbose` and `--backends-help` flags, pass through to `run_profile()` |
| `tests/test_profiler.py` | Modify | Add tests for new formatting functions and CLI flags |

---

## Chunk 1: Compact Output Formatter

### Task 1: Add `format_dims` helper

**Files:**
- Modify: `src/network_estimation/profiler.py` (after line 115, before `CorrectnessResult`)
- Test: `tests/test_profiler.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_profiler.py, add at top:
from network_estimation.profiler import format_dims

class TestFormatDims:
    def test_small_number(self) -> None:
        assert format_dims(64, 4, 500) == "64×4×500"

    def test_thousands(self) -> None:
        assert format_dims(64, 4, 10_000) == "64×4×10k"

    def test_hundreds_of_thousands(self) -> None:
        assert format_dims(256, 8, 100_000) == "256×8×100k"

    def test_millions(self) -> None:
        assert format_dims(256, 8, 1_000_000) == "256×8×1M"

    def test_large_millions(self) -> None:
        assert format_dims(256, 8, 16_700_000) == "256×8×16.7M"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/test_profiler.py::TestFormatDims -v`
Expected: FAIL with ImportError (format_dims not defined)

- [ ] **Step 3: Write minimal implementation**

Add to `src/network_estimation/profiler.py` after line 115 (after PRESETS dict):

```python
def format_dims(width: int, depth: int, n_samples: int) -> str:
    """Format dimensions as compact 'w×d×n' string with k/M suffixes."""
    if n_samples >= 1_000_000:
        if n_samples % 1_000_000 == 0:
            n_str = f"{n_samples // 1_000_000}M"
        else:
            n_str = f"{n_samples / 1_000_000:.1f}M"
    elif n_samples >= 1_000:
        if n_samples % 1_000 == 0:
            n_str = f"{n_samples // 1_000}k"
        else:
            n_str = f"{n_samples / 1_000:.1f}k"
    else:
        n_str = str(n_samples)
    return f"{width}×{depth}×{n_str}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.10 -m pytest tests/test_profiler.py::TestFormatDims -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/profiler.py tests/test_profiler.py
git commit -m "feat: add format_dims helper for compact dimension display"
```

---

### Task 2: Add `format_compact_output` function

**Files:**
- Modify: `src/network_estimation/profiler.py` (add new function after `format_terminal_table`)
- Test: `tests/test_profiler.py`

- [ ] **Step 1: Write the failing test**

```python
from network_estimation.profiler import (
    CorrectnessResult,
    TimingResult,
    format_compact_output,
)

class TestFormatCompactOutput:
    def _make_results(self):
        """Build minimal test data with two backends."""
        correctness = [
            CorrectnessResult(backend_name="numpy", passed=True),
            CorrectnessResult(backend_name="scipy", passed=True),
        ]
        timing = [
            TimingResult(
                backend_name="numpy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0001], median_time=0.0001, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0002], median_time=0.0002, speedup_vs_numpy=0.5,
            ),
            TimingResult(
                backend_name="numpy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
            ),
        ]
        skipped = {"pytorch": "pip install torch>=2.0"}
        hardware = {
            "platform": "macOS-26.3-arm64",
            "machine": "arm64",
            "cpu_count_physical": 16,
            "cpu_count_logical": 16,
            "ram_total_bytes": 64 * 1024**3,
            "python_version": "3.10.17",
            "numpy_version": "2.2.6",
            "os": "Darwin",
        }
        return correctness, timing, skipped, hardware

    def test_contains_hardware_context_line(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "arm64" in output
        assert "16 cores" in output
        assert "64.0 GB" in output

    def test_contains_skipped_one_liner(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Skipped:" in output
        assert "pytorch" in output
        assert "--backends-help" in output

    def test_no_skipped_line_when_none_skipped(self) -> None:
        cr, tr, _, hw = self._make_results()
        output = format_compact_output(cr, tr, {}, hardware_info=hw)
        assert "Skipped:" not in output

    def test_contains_leaderboard(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "#1" in output
        assert "#2" in output
        assert "Leaderboard" in output

    def test_contains_detail_table(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Detail" in output

    def test_contains_verbose_hint(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "--verbose" in output

    def test_single_backend_omits_leaderboard(self) -> None:
        cr = [CorrectnessResult(backend_name="numpy", passed=True)]
        tr = [
            TimingResult(
                backend_name="numpy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0001], median_time=0.0001, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="numpy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
            ),
        ]
        output = format_compact_output(cr, tr, {})
        assert "Leaderboard" not in output

    def test_zero_passed_backends(self) -> None:
        cr = [CorrectnessResult(backend_name="numpy", passed=False, error="boom")]
        output = format_compact_output(cr, [], {})
        assert "No backends passed" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/test_profiler.py::TestFormatCompactOutput -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

Add `format_compact_output` to `src/network_estimation/profiler.py` after the existing `format_terminal_table` function (after line 535). This is the core of the redesign.

The function signature:

```python
def format_compact_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    hardware_info: Optional[Dict[str, Any]] = None,
) -> str:
```

Implementation details:

**Zone 1 — Context lines:**
- Hardware: `{os} {machine} · {cores} cores · {ram} GB · Python {ver} · NumPy {ver}`
  - Cores: use `cpu_count_physical`, fall back to `cpu_count_logical`, omit if both None
  - RAM: `ram_total_bytes / (1024**3)` formatted to 1 decimal
- Skipped: `Skipped: name1, name2 (--backends-help for install info)` — omit line if empty
- Correctness: `Correctness: numpy ✓  scipy ✓` — use `⚠ FAIL: name (use --verbose for details)` for failures

**Zone 2 — Leaderboard:**
- Group `run_mlp` results by `(width, depth, n_samples)` tuple
- Within each group, sort by `median_time` ascending
- Baseline: numpy if present, else fastest
- Show `#N  backend  time` with speedup multiplier for non-baseline (green >1x, red <1x)
- Omit zone entirely if only 1 backend passed

**Zone 3 — Detail table (Rich table):**
- Columns: Backend, Dims, run_mlp, output_stats, Matmul%, ReLU%, Overhead%, status
- One row per backend per dim combo
- For Matmul/ReLU: use Unicode block chars as mini-bars: `[blue]████[/]` proportional to percentage (max 5 blocks)
- Group dim combos with separator

**Footer:** `Use --verbose for full timing tables with raw times and per-layer breakdowns`

Use `shutil.get_terminal_size().columns` (fallback 120) for Rich Console width.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.10 -m pytest tests/test_profiler.py::TestFormatCompactOutput -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/profiler.py tests/test_profiler.py
git commit -m "feat: add format_compact_output for leaderboard + detail view"
```

---

### Task 3: Rename `format_terminal_table` → `format_verbose_output`

**Files:**
- Modify: `src/network_estimation/profiler.py` (line 404)
- Modify: `tests/test_profiler.py` (any references)

- [ ] **Step 1: Rename function**

In `src/network_estimation/profiler.py`, rename `format_terminal_table` to `format_verbose_output` on line 404. This is only called from `run_profile()` on line 750, so update that call site too.

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `python3.10 -m pytest tests/test_profiler.py -v`
Expected: PASS (existing tests don't directly call `format_terminal_table`)

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/profiler.py
git commit -m "refactor: rename format_terminal_table to format_verbose_output"
```

---

## Chunk 2: Wire Up CLI Flags and run_profile

### Task 4: Add `verbose` parameter to `run_profile()`

**Files:**
- Modify: `src/network_estimation/profiler.py:582-765`
- Test: `tests/test_profiler.py`

- [ ] **Step 1: Write the failing test**

```python
class TestRunProfileVerbose:
    def test_default_uses_compact_format(self) -> None:
        """Default (verbose=False) should use compact leaderboard format."""
        output, _ = run_profile(
            preset_name="super-quick", backend_filter=["numpy"]
        )
        assert "Leaderboard" in output or "Detail" in output
        # Should NOT contain the old-style verbose headers
        assert "Timing Results" not in output

    def test_verbose_includes_both_compact_and_full(self) -> None:
        """verbose=True should show compact output PLUS full tables."""
        output, _ = run_profile(
            preset_name="super-quick", backend_filter=["numpy"], verbose=True
        )
        # Compact content present
        assert "Detail" in output
        # Full verbose tables also present
        assert "Timing Results" in output

    def test_multi_dim_leaderboard_grouping(self) -> None:
        """Multiple dimension combos should produce separate leaderboard groups."""
        output, _ = run_profile(
            preset_name="quick", backend_filter=["numpy", "scipy"], verbose=False
        )
        # quick preset has 2 depths × 2 n_samples = 4 combos
        # Each gets a leaderboard group header
        assert "Leaderboard" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/test_profiler.py::TestRunProfileVerbose -v`
Expected: FAIL (run_profile doesn't accept `verbose` yet)

- [ ] **Step 3: Add `verbose` parameter to `run_profile()`**

In `src/network_estimation/profiler.py`, modify `run_profile()`:

1. Add `verbose: bool = False` parameter (line 587, after `max_threads`)
2. On line 750, change the formatting call:

```python
# Replace:
terminal_output = format_terminal_table(
    correctness_results, timing_results, skipped,
    hardware_info=hardware_info,
)

# With (compact first, verbose appended when requested — per spec):
terminal_output = format_compact_output(
    correctness_results, timing_results, skipped,
    hardware_info=hardware_info,
)
if verbose:
    terminal_output += "\n" + format_verbose_output(
        correctness_results, timing_results, skipped,
        hardware_info=hardware_info,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.10 -m pytest tests/test_profiler.py::TestRunProfileVerbose -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/profiler.py tests/test_profiler.py
git commit -m "feat: add verbose parameter to run_profile()"
```

---

### Task 5: Add `--verbose` and `--backends-help` CLI flags

**Files:**
- Modify: `src/network_estimation/cli.py:502-529` (argument parser)
- Modify: `src/network_estimation/cli.py:886-899` (handler)
- Test: `tests/test_profiler.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestCLIFlags:
    def test_verbose_flag_accepted(self) -> None:
        """CLI should accept --verbose flag."""
        from network_estimation.cli import _build_participant_parser
        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick", "--verbose"])
        assert args.verbose is True

    def test_verbose_flag_default_false(self) -> None:
        from network_estimation.cli import _build_participant_parser
        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick"])
        assert args.verbose is False

    def test_backends_help_flag_accepted(self) -> None:
        from network_estimation.cli import _build_participant_parser
        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--backends-help"])
        assert args.backends_help is True

    def test_backends_help_prints_and_exits(self) -> None:
        """--backends-help should print install info and return 0 without profiling."""
        from network_estimation.cli import _main_participant
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _main_participant(["profile-simulation", "--backends-help"])
        assert rc == 0
        output = buf.getvalue()
        # Should mention at least one backend (either install hint or "All backends")
        assert "install" in output.lower() or "All backends" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/test_profiler.py::TestCLIFlags -v`
Expected: FAIL

- [ ] **Step 3: Add flags to parser and handler**

In `src/network_estimation/cli.py`:

**Parser** (after line 529, before `return parser`):

```python
profile_parser.add_argument(
    "--verbose", action="store_true", default=False,
    help="Show full timing tables with all columns and raw data.",
)
profile_parser.add_argument(
    "--backends-help", action="store_true", default=False,
    help="Print install instructions for all backends and exit.",
)
```

**Handler** (replace lines 886-899):

```python
if command == "profile-simulation":
    from .profiler import run_profile
    from .simulation_backends import ALL_BACKEND_NAMES, INSTALL_HINTS, get_available_backends

    # Handle --backends-help early exit
    if getattr(args, "backends_help", False):
        available = get_available_backends()
        skipped = {
            name: INSTALL_HINTS.get(name, "")
            for name in ALL_BACKEND_NAMES
            if name not in available
        }
        if skipped:
            for name, hint in skipped.items():
                print(f"  {name}: {hint}")
        else:
            print("All backends are installed.")
        return 0

    backend_filter = None
    if args.backends:
        backend_filter = [b.strip() for b in args.backends.split(",")]
    terminal_output, _ = run_profile(
        preset_name=str(args.preset),
        backend_filter=backend_filter,
        output_path=args.output,
        show_progress=not json_output,
        max_threads=args.max_threads,
        verbose=bool(args.verbose),
    )
    print(terminal_output)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.10 -m pytest tests/test_profiler.py::TestCLIFlags -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/cli.py tests/test_profiler.py
git commit -m "feat: add --verbose and --backends-help CLI flags"
```

---

### Task 6: Update existing tests and run full suite

**Files:**
- Modify: `tests/test_profiler.py`

- [ ] **Step 1: Update existing tests that check for old output format**

The test `test_quick_preset_runs` on line 29-34 checks for `"Timing Results"` in output. Since default output is now compact, update this:

```python
def test_quick_preset_runs(self) -> None:
    terminal_output, _ = run_profile(
        preset_name="super-quick", backend_filter=["numpy"]
    )
    assert "numpy" in terminal_output
    assert "Detail" in terminal_output  # compact format
```

- [ ] **Step 2: Run the full test suite**

Run: `python3.10 -m pytest tests/test_profiler.py -v`
Expected: All PASS

- [ ] **Step 3: Manual verification**

Run the profiler to visually verify the output:

```bash
python3.10 -c "from network_estimation.cli import main; main(['profile-simulation', '--preset', 'super-quick'])"
```

Verify:
- Zone 1 shows condensed hardware on one line
- Zone 2 shows leaderboard (or is omitted for single backend)
- Zone 3 shows compact detail table
- Footer shows `--verbose` hint

Then verify verbose mode:
```bash
python3.10 -c "from network_estimation.cli import main; main(['profile-simulation', '--preset', 'super-quick', '--verbose'])"
```

Verify: compact output is shown PLUS the full tables below it.

- [ ] **Step 4: Commit**

```bash
git add tests/test_profiler.py
git commit -m "test: update profiler tests for compact output format"
```
