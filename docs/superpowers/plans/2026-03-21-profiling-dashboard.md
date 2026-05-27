# Profiling Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a self-contained interactive HTML dashboard from profiling results, reusing the network-explorer design system.

**Architecture:** A Python script (`profiling/generate_dashboard.py`) loads profiling JSON, normalizes single/multi-config formats, extracts CSS from `tools/network-explorer/src/App.css`, fetches+caches React/Recharts CDN libs, and assembles a single HTML file with embedded React components (using `React.createElement`, no JSX/Babel). The React app renders a speedup heatmap, click-to-open detail modal, CPU scaling chart, and sortable data table.

**Tech Stack:** Python 3.8+, React 18, Recharts 2 (UMD via CDN, inlined at gen time), CSS variables from network-explorer

**Spec:** `docs/superpowers/specs/2026-03-21-profiling-dashboard-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `profiling/generate_dashboard.py` | CLI entry point: arg parsing, data loading/normalization, CSS extraction, lib caching, HTML assembly, file output |
| `profiling/dashboard_components.js` | All React components as a JS string template (App, Header, SpeedupHeatmap, CellDetailModal, CPUScalingChart, DataTable) using `React.createElement` |
| `profiling/dashboard_styles.css` | Dashboard-specific CSS (heatmap colors, modal overlay, backend colors, table styles) that gets appended after network-explorer base CSS |
| `profiling/tests/test_generate_dashboard.py` | Tests for data normalization, CSS extraction, HTML generation, CLI args |

---

## Chunk 1: Data Loading & Normalization

### Task 1: Data normalization functions

**Files:**
- Create: `profiling/generate_dashboard.py`
- Create: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write test for multi-config loading**

```python
# profiling/tests/test_generate_dashboard.py
"""Tests for dashboard generation."""
import json
import os
import tempfile
from profiling.generate_dashboard import load_data, normalize_data

MULTI_CONFIG_DATA = {
    "run_id": "2026-03-20-test",
    "git_commit": "abc1234",
    "git_dirty": True,
    "collected_at": "2026-03-20T21:00:00+00:00",
    "configs": {
        "compute-small": {
            "hardware": {"cpu_count_logical": 2, "ram_total_bytes": 2147483648,
                         "hostname": "ip-test", "platform": "Linux"},
            "backend_versions": {"numpy": "1.24.0", "scipy": "1.10.0"},
            "skipped_backends": {"numba": "No numba installed"},
            "correctness": [
                {"backend": "numpy", "passed": True, "error": ""},
                {"backend": "scipy", "passed": False,
                 "error": "max_diff=0.5 exceeds threshold"},
            ],
            "timing": [
                {"backend": "numpy", "operation": "run_mlp", "width": 256,
                 "depth": 4, "n_samples": 10000, "times": [0.05, 0.052, 0.051],
                 "median_time": 0.051, "speedup_vs_numpy": 1.0},
                {"backend": "scipy", "operation": "run_mlp", "width": 256,
                 "depth": 4, "n_samples": 10000, "times": [0.03, 0.031, 0.032],
                 "median_time": 0.031, "speedup_vs_numpy": 1.645},
            ],
        },
        "empty-config": {
            "hardware": {"cpu_count_logical": 1, "ram_total_bytes": 1073741824,
                         "hostname": "ip-empty", "platform": "Linux"},
            "backend_versions": {"numpy": "1.24.0"},
            "skipped_backends": {},
            "correctness": [],
            "timing": [],
        },
    },
}


def test_load_multi_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MULTI_CONFIG_DATA, f)
        path = f.name
    try:
        data = load_data(path)
        assert data["run_id"] == "2026-03-20-test"
        assert "compute-small" in data["configs"]
        assert len(data["configs"]["compute-small"]["timing"]) == 2
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py::test_load_multi_config -v`
Expected: FAIL — `ImportError: cannot import name 'load_data'`

- [ ] **Step 3: Write test for single-config normalization**

```python
# append to profiling/tests/test_generate_dashboard.py

SINGLE_CONFIG_DATA = {
    "hardware": {"cpu_count_logical": 8, "ram_total_bytes": 17179869184,
                 "hostname": "my-macbook", "platform": "Darwin"},
    "backend_versions": {"numpy": "1.24.0"},
    "skipped_backends": {},
    "correctness": [{"backend": "numpy", "passed": True, "error": ""}],
    "timing": [
        {"backend": "numpy", "operation": "run_mlp", "width": 256,
         "depth": 4, "n_samples": 10000, "times": [0.05, 0.052],
         "median_time": 0.051, "speedup_vs_numpy": 1.0},
    ],
}


def test_normalize_single_config():
    normalized = normalize_data(SINGLE_CONFIG_DATA)
    assert "configs" in normalized
    assert "run_id" in normalized
    # Config name derived from hostname
    config_names = list(normalized["configs"].keys())
    assert len(config_names) == 1
    assert normalized["configs"][config_names[0]]["hardware"]["cpu_count_logical"] == 8


def test_normalize_multi_config_passthrough():
    normalized = normalize_data(MULTI_CONFIG_DATA)
    assert normalized["run_id"] == "2026-03-20-test"
    assert "compute-small" in normalized["configs"]
```

- [ ] **Step 4: Implement load_data and normalize_data**

```python
# profiling/generate_dashboard.py
"""Generate self-contained HTML profiling dashboard."""
import argparse
import json
import os
import sys


def load_data(path):
    """Load profiling JSON from file path."""
    with open(path) as f:
        return json.load(f)


def normalize_data(data):
    """Normalize single-config or multi-config data into multi-config format."""
    if "configs" in data:
        return data
    # Single-config: wrap into multi-config format
    hostname = data.get("hardware", {}).get("hostname", "local")
    config_name = hostname if hostname != "local" else "local"
    return {
        "run_id": f"local-{config_name}",
        "git_commit": "",
        "git_dirty": False,
        "collected_at": "",
        "configs": {config_name: data},
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add profiling/generate_dashboard.py profiling/tests/test_generate_dashboard.py
git commit -m "feat(dashboard): add data loading and normalization"
```

---

### Task 2: CLI argument parsing

**Files:**
- Modify: `profiling/generate_dashboard.py`
- Modify: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write test for CLI args**

```python
# append to profiling/tests/test_generate_dashboard.py

from profiling.generate_dashboard import parse_args


def test_parse_args_run_id():
    args = parse_args(["--run-id", "2026-03-20-test"])
    assert args.run_id == "2026-03-20-test"
    assert args.input is None


def test_parse_args_input():
    args = parse_args(["--input", "output.json"])
    assert args.input == "output.json"
    assert args.run_id is None


def test_parse_args_output():
    args = parse_args(["--input", "output.json", "--output", "my.html"])
    assert args.output == "my.html"


def test_parse_args_requires_one():
    import pytest
    with pytest.raises(SystemExit):
        parse_args([])
```

- [ ] **Step 1b: Write test for resolve_paths**

```python
# append to profiling/tests/test_generate_dashboard.py

from profiling.generate_dashboard import resolve_paths


def test_resolve_paths_run_id():
    args = parse_args(["--run-id", "my-run"])
    input_path, output_path = resolve_paths(args)
    assert input_path == os.path.join("profiling", "results", "my-run", "combined.json")
    assert output_path == os.path.join("profiling", "results", "my-run", "dashboard.html")


def test_resolve_paths_input_default_output():
    args = parse_args(["--input", "output.json"])
    input_path, output_path = resolve_paths(args)
    assert input_path == "output.json"
    assert output_path == "dashboard.html"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py::test_parse_args_run_id -v`
Expected: FAIL — `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Implement parse_args**

```python
# add to profiling/generate_dashboard.py

def parse_args(argv=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate profiling dashboard HTML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Load from profiling/results/{run-id}/combined.json")
    group.add_argument("--input", help="Path to combined.json or single-config output.json")
    parser.add_argument("--output", help="Output HTML path")
    return parser.parse_args(argv)


def resolve_paths(args):
    """Resolve input/output file paths from parsed args."""
    if args.run_id:
        input_path = os.path.join("profiling", "results", args.run_id, "combined.json")
        default_output = os.path.join("profiling", "results", args.run_id, "dashboard.html")
    else:
        input_path = args.input
        default_output = "dashboard.html"
    output_path = args.output or default_output
    return input_path, output_path
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add profiling/generate_dashboard.py profiling/tests/test_generate_dashboard.py
git commit -m "feat(dashboard): add CLI argument parsing"
```

---

## Chunk 2: CSS Extraction & Lib Caching

### Task 3: CSS extraction from network-explorer

**Files:**
- Modify: `profiling/generate_dashboard.py`
- Create: `profiling/dashboard_styles.css`
- Modify: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write test for CSS extraction**

```python
# append to profiling/tests/test_generate_dashboard.py

from profiling.generate_dashboard import extract_base_css


def test_extract_base_css():
    css = extract_base_css()
    # Must contain CSS variables
    assert "--coral:" in css
    assert "--gray-900:" in css
    assert "--font-sans:" in css
    # Must contain Google Fonts import
    assert "@import url(" in css
    # Must contain .app-header styles
    assert ".app-header {" in css or ".app-header{" in css
    assert ".app-header h1" in css
    # Must NOT contain component-specific class rules
    assert ".sidebar {" not in css
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py::test_extract_base_css -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement extract_base_css**

```python
# add to profiling/generate_dashboard.py

def extract_base_css():
    """Extract base CSS variables and resets from network-explorer App.css."""
    # Find App.css relative to this repo
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(repo_root, "tools", "network-explorer", "src", "App.css")
    with open(css_path) as f:
        full_css = f.read()

    # Extract everything before the App Shell section
    marker = "/* ──────────── App Shell ──────────── */"
    idx = full_css.find(marker)
    if idx == -1:
        # Fallback: take first 2000 chars (variables + resets)
        base = full_css[:2000]
    else:
        base = full_css[:idx]

    # Also extract .app-header styles
    header_start = full_css.find(".app-header {")
    if header_start != -1:
        # Find the closing brace for .app-header and .app-header h1
        header_section = ""
        pos = header_start
        brace_count = 0
        blocks_found = 0
        while pos < len(full_css) and blocks_found < 2:
            if full_css[pos] == "{":
                brace_count += 1
            elif full_css[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    blocks_found += 1
                    header_section = full_css[header_start:pos + 1]
                    # Check if next block is .app-header h1
                    next_chunk = full_css[pos + 1:pos + 50].strip()
                    if next_chunk.startswith(".app-header h1"):
                        header_start_h1 = full_css.find(".app-header h1", pos)
                        pos2 = full_css.find("}", header_start_h1)
                        header_section = full_css[header_start:pos2 + 1]
                    break
            pos += 1
        base += "\n" + header_section

    return base
```

- [ ] **Step 4: Create dashboard-specific CSS file**

```css
/* profiling/dashboard_styles.css */
/* Dashboard-specific styles — appended after network-explorer base CSS */

.dashboard { min-height: 100vh; background: var(--gray-50); }

/* Heatmap */
.heatmap-section { padding: 20px 28px; }
.section-header {
  font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--gray-400);
  font-weight: 600; margin-bottom: 12px;
}
.heatmap-filters { display: flex; gap: 10px; margin-bottom: 14px; }
.filter-select {
  background: var(--gray-100); border: 1px solid var(--gray-200);
  padding: 5px 10px; border-radius: 6px; font-size: 11px;
  font-family: var(--font-sans); color: var(--gray-900);
}
.filter-select:focus {
  outline: none; border-color: var(--coral);
  box-shadow: 0 0 0 3px var(--coral-light);
}
.heatmap-table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  text-align: center; font-size: 12px;
  border: 1px solid var(--gray-200); border-radius: var(--radius);
  overflow: hidden;
}
.heatmap-table th {
  padding: 8px; font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--gray-400);
  font-weight: 600; background: var(--gray-50);
  border-bottom: 1px solid var(--gray-200);
}
.heatmap-table th:first-child { text-align: left; padding-left: 12px; }
.heatmap-table td {
  padding: 8px; font-family: var(--font-mono); font-size: 11px;
  border-bottom: 1px solid var(--gray-100); cursor: pointer;
}
.heatmap-table td:first-child {
  text-align: left; padding-left: 12px; font-family: var(--font-sans);
  font-weight: 500; font-size: 12px; cursor: default;
}
.heatmap-table tr:nth-child(even) { background: var(--gray-50); }
.heatmap-table td.best { font-weight: 600; }
.config-hw { font-size: 10px; color: var(--gray-400); font-weight: 400; }
.cell-warning { color: var(--warning); cursor: help; }

/* Modal */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.3);
  display: flex; align-items: center; justify-content: center; z-index: 100;
}
.modal {
  background: var(--white); border: 1px solid var(--gray-200);
  border-radius: var(--radius); box-shadow: 0 8px 24px rgba(0,0,0,0.12);
  width: 560px; max-height: 90vh; overflow-y: auto;
}
.modal-header {
  padding: 14px 20px; display: flex; justify-content: space-between;
  align-items: center; border-bottom: 1px solid var(--gray-200);
}
.modal-title {
  font-family: var(--font-accent); font-weight: 700;
  font-size: 16px; color: var(--gray-900);
}
.badge-faster {
  background: var(--success); color: white; padding: 3px 10px;
  border-radius: var(--radius-pill); font-size: 11px; font-weight: 600;
}
.badge-slower {
  background: var(--coral); color: white; padding: 3px 10px;
  border-radius: var(--radius-pill); font-size: 11px; font-weight: 600;
}
.modal-close {
  color: var(--gray-400); cursor: pointer; font-size: 16px;
  border: none; background: none; padding: 4px 8px;
}
.modal-close:hover { color: var(--gray-900); }
.modal-filters {
  padding: 10px 20px; background: var(--gray-50);
  display: flex; gap: 10px; border-bottom: 1px solid var(--gray-200);
}
.filter-label {
  font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--gray-400); font-weight: 600; margin-bottom: 3px;
}
.modal-body {
  padding: 16px 20px; display: grid;
  grid-template-columns: 1fr 1fr; gap: 20px;
}
.modal-footer {
  padding: 10px 20px 14px; background: var(--gray-50);
  border-top: 1px solid var(--gray-200);
  display: flex; align-items: center; gap: 10px; font-size: 11px;
}

/* Timing bars */
.timing-big {
  font-family: var(--font-accent); font-weight: 700;
  font-size: 28px; color: var(--gray-900);
}
.timing-unit { font-size: 14px; color: var(--gray-400); font-weight: 400; }
.timing-sub { color: var(--gray-400); font-size: 11px; margin-bottom: 14px; }
.run-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.run-bar-label {
  color: var(--gray-400); font-size: 10px; width: 28px;
  font-family: var(--font-mono);
}
.run-bar-track {
  flex: 1; background: var(--gray-100); border-radius: 4px;
  height: 16px; overflow: hidden;
}
.run-bar-fill {
  height: 100%; border-radius: 4px; display: flex;
  align-items: center; padding-left: 8px;
}
.run-bar-value {
  font-size: 10px; color: white; font-family: var(--font-mono);
}
.timing-stats {
  margin-top: 8px; color: var(--gray-400); font-size: 10px;
  font-family: var(--font-mono);
}

/* Breakdown */
.breakdown-bar {
  height: 22px; border-radius: 6px; overflow: hidden;
  display: flex; margin-bottom: 12px;
}
.breakdown-bar span {
  font-size: 10px; color: white; font-weight: 500;
  display: flex; align-items: center; justify-content: center;
}
.breakdown-legend { display: flex; flex-direction: column; gap: 8px; font-size: 12px; }
.breakdown-row { display: flex; justify-content: space-between; }
.breakdown-swatch {
  width: 10px; height: 10px; border-radius: 2px; display: inline-block;
  margin-right: 6px; vertical-align: middle;
}
.breakdown-value {
  font-weight: 600; font-family: var(--font-mono); font-size: 11px;
}
.breakdown-pct { color: var(--gray-400); font-size: 10px; margin-left: 4px; }

/* Sparkline */
.sparkline { display: flex; align-items: flex-end; gap: 1px; height: 28px; margin-top: 14px; }
.sparkline-bar {
  flex: 1; background: var(--chart-3); border-radius: 1px 1px 0 0;
}
.sparkline-labels {
  display: flex; justify-content: space-between; margin-top: 2px;
  font-size: 8px; color: var(--gray-400); font-family: var(--font-mono);
}

/* Comparison footer */
.compare-label {
  color: var(--gray-400); font-size: 9px; text-transform: uppercase;
  letter-spacing: 0.5px; font-weight: 600;
}
.compare-bar {
  flex: 1; background: var(--gray-100); border-radius: 4px;
  height: 16px; overflow: hidden;
}
.compare-fill {
  height: 100%; border-radius: 4px; display: flex;
  align-items: center; justify-content: center;
}
.compare-fill span {
  font-size: 9px; color: white; font-family: var(--font-mono);
}

/* Scaling chart section */
.scaling-section { padding: 20px 28px; }
.scaling-chart-container {
  background: var(--white); border: 1px solid var(--gray-200);
  border-radius: var(--radius); padding: 16px;
}

/* Data table section */
.datatable-section { padding: 20px 28px; }
.datatable {
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-size: 11px; border: 1px solid var(--gray-200);
  border-radius: var(--radius); overflow: hidden;
}
.datatable th {
  padding: 6px 8px; font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--gray-400); font-weight: 600;
  background: var(--gray-50); border-bottom: 1px solid var(--gray-200);
  cursor: pointer; user-select: none;
}
.datatable th:hover { color: var(--gray-900); }
.datatable th .sort-arrow { margin-left: 4px; font-size: 8px; }
.datatable td {
  padding: 4px 8px; font-family: var(--font-mono);
  border-bottom: 1px solid var(--gray-100);
}
.datatable tr:nth-child(even) { background: var(--gray-50); }
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add profiling/generate_dashboard.py profiling/dashboard_styles.css profiling/tests/test_generate_dashboard.py
git commit -m "feat(dashboard): add CSS extraction and dashboard-specific styles"
```

---

### Task 4: CDN library fetching and caching

**Files:**
- Modify: `profiling/generate_dashboard.py`
- Modify: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write test for lib fetching**

```python
# append to profiling/tests/test_generate_dashboard.py

from profiling.generate_dashboard import fetch_cdn_libs

def test_fetch_cdn_libs_returns_dict():
    libs = fetch_cdn_libs()
    assert "react" in libs
    assert "react-dom" in libs
    assert "recharts" in libs
    # Each value should be a non-empty string (JS source)
    for name, source in libs.items():
        assert len(source) > 1000, f"{name} source too small"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py::test_fetch_cdn_libs_returns_dict -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement fetch_cdn_libs**

```python
# add to profiling/generate_dashboard.py
import urllib.request

CDN_LIBS = {
    "react": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "recharts": "https://unpkg.com/recharts@2/umd/Recharts.min.js",
}


def fetch_cdn_libs(cache_dir=None):
    """Fetch CDN libraries, caching locally for subsequent runs."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lib-cache")
    os.makedirs(cache_dir, exist_ok=True)

    libs = {}
    for name, url in CDN_LIBS.items():
        cache_path = os.path.join(cache_dir, f"{name}.min.js")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                libs[name] = f.read()
        else:
            print(f"Fetching {name} from {url}...")
            resp = urllib.request.urlopen(url)
            source = resp.read().decode("utf-8")
            with open(cache_path, "w") as f:
                f.write(source)
            libs[name] = source
    return libs
```

- [ ] **Step 4: Add `.lib-cache/` to `.gitignore`**

Append `profiling/.lib-cache/` to the project's `.gitignore`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add profiling/generate_dashboard.py profiling/tests/test_generate_dashboard.py .gitignore
git commit -m "feat(dashboard): add CDN library fetching with local cache"
```

---

## Chunk 3: React Components

### Task 5: React component JS file

**Files:**
- Create: `profiling/dashboard_components.js`

This is the largest task. The JS file contains all React components using `React.createElement` (aliased as `h`). All components read data from `window.__PROFILING_DATA__`.

- [ ] **Step 1: Write utility functions and App component**

Create `profiling/dashboard_components.js`:

```javascript
// profiling/dashboard_components.js
// All React components for the profiling dashboard.
// Uses React.createElement (no JSX/Babel). Data from window.__PROFILING_DATA__.
'use strict';

var h = React.createElement;
var useState = React.useState;
var useEffect = React.useEffect;

var BACKEND_COLORS = {
  numpy: '#4A90D9', pytorch: '#e65100', jax: '#334155',
  cython: '#23B761', numba: '#6a1b9a', scipy: '#00838f'
};
var BACKEND_ORDER = ['numpy', 'pytorch', 'jax', 'cython', 'numba', 'scipy'];

function getSpeedupColor(s) {
  if (s == null) return 'transparent';
  if (s >= 0.9 && s <= 1.1) return '#F8F9F9';
  if (s > 1.1) {
    var t = Math.min((s - 1.1) / 3.0, 1.0);
    var r = Math.round(35 + (1 - t) * 220);
    var g = Math.round(183 + (1 - t) * 72);
    var b = Math.round(97 + (1 - t) * 152);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  var t2 = Math.min((1.0 - s) / 0.5, 1.0);
  var r2 = Math.round(240 + (1 - t2) * 8);
  var g2 = Math.round(82 + (1 - t2) * 165);
  var b2 = Math.round(77 + (1 - t2) * 172);
  return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
}

function formatTime(s) {
  if (s < 0.001) return (s * 1e6).toFixed(0) + 'µs';
  if (s < 1) return (s * 1000).toFixed(1) + 'ms';
  return s.toFixed(3) + 's';
}

function uniqueVals(timing, key) {
  var s = {};
  timing.forEach(function(t) { s[t[key]] = true; });
  return Object.keys(s).map(function(v) { return isNaN(v) ? v : Number(v); }).sort(function(a, b) { return a - b; });
}

function getUnique(data, key) {
  var all = [];
  Object.values(data.configs).forEach(function(c) { all = all.concat(c.timing); });
  return uniqueVals(all, key);
}

function App() {
  var data = window.__PROFILING_DATA__;
  var operations = getUnique(data, 'operation');
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');
  var isMulti = Object.keys(data.configs).length > 1;

  var s = useState({
    operation: operations[0] || 'run_mlp',
    width: widths[widths.length - 1] || 256,
    depth: depths[depths.length - 1] || 4,
    nSamples: nSamples[nSamples.length - 1] || 10000,
    modal: null
  });
  var filters = s[0]; var setFilters = s[1];

  function onFilterChange(key, val) {
    setFilters(function(prev) {
      var n = {}; for (var k in prev) n[k] = prev[k];
      n[key] = isNaN(val) ? val : Number(val);
      return n;
    });
  }

  return h('div', {className: 'dashboard'},
    h(Header, {data: data}),
    h(SpeedupHeatmap, {data: data, filters: filters, onFilterChange: onFilterChange,
      onCellClick: function(m) { onFilterChange('modal', m); }}),
    isMulti ? h(CPUScalingChart, {data: data, filters: filters, onFilterChange: onFilterChange}) : null,
    h(DataTable, {data: data}),
    filters.modal ? h(CellDetailModal, {data: data, config: filters.modal.config,
      backend: filters.modal.backend, filters: filters, onFilterChange: onFilterChange,
      onClose: function() { onFilterChange('modal', null); }}) : null
  );
}
```

- [ ] **Step 2: Write Header component**

```javascript
// append to profiling/dashboard_components.js

function Header(props) {
  var d = props.data;
  return h('header', {className: 'app-header'},
    h('h1', null, 'nestim Profiling Dashboard'),
    h('div', {style: {display: 'flex', alignItems: 'center', gap: '12px',
      fontSize: '12px', color: '#6B7280'}},
      d.run_id ? h('span', null, d.run_id) : null,
      d.git_commit ? h('span', {style: {fontFamily: 'var(--font-mono)'}},
        d.git_commit.substring(0, 7),
        d.git_dirty ? h('span', {className: 'badge-slower',
          style: {marginLeft: '4px', fontSize: '9px', padding: '1px 6px'}}, 'dirty') : null
      ) : null,
      d.collected_at ? h('span', null, new Date(d.collected_at).toLocaleString()) : null
    )
  );
}
```

- [ ] **Step 3: Write SpeedupHeatmap component**

```javascript
// append to profiling/dashboard_components.js

function SpeedupHeatmap(props) {
  var data = props.data, f = props.filters;
  var configs = Object.keys(data.configs);
  var backends = BACKEND_ORDER.filter(function(b) {
    return configs.some(function(c) {
      return data.configs[c].timing.some(function(t) { return t.backend === b; }) ||
        (data.configs[c].skipped_backends && data.configs[c].skipped_backends[b]);
    });
  });

  function getCell(config, backend) {
    var cfg = data.configs[config];
    if (cfg.skipped_backends && cfg.skipped_backends[backend]) return {type: 'skipped', reason: cfg.skipped_backends[backend]};
    var entry = cfg.timing.find(function(t) {
      return t.backend === backend && t.operation === f.operation &&
        t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
    });
    if (!entry) return {type: 'missing'};
    var corr = cfg.correctness.find(function(c) { return c.backend === backend; });
    return {type: 'data', speedup: entry.speedup_vs_numpy,
      correctnessFailed: corr && !corr.passed, correctnessError: corr ? corr.error : ''};
  }

  var operations = getUnique(data, 'operation');
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');

  function sel(label, key, opts, val) {
    return h('div', null,
      h('div', {className: 'filter-label'}, label),
      h('select', {className: 'filter-select', value: val,
        onChange: function(e) { props.onFilterChange(key, e.target.value); }},
        opts.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
      )
    );
  }

  return h('section', {className: 'heatmap-section'},
    h('div', {className: 'section-header'}, 'SPEEDUP VS NUMPY'),
    h('div', {className: 'heatmap-filters'},
      sel('Operation', 'operation', operations, f.operation),
      sel('Width', 'width', widths, f.width),
      sel('Depth', 'depth', depths, f.depth),
      sel('N Samples', 'nSamples', nSamples, f.nSamples)
    ),
    h('table', {className: 'heatmap-table'},
      h('thead', null, h('tr', null,
        h('th', null, 'Config'),
        backends.map(function(b) { return h('th', {key: b}, b); })
      )),
      h('tbody', null, configs.map(function(config) {
        var hw = data.configs[config].hardware;
        var ramGB = Math.round((hw.ram_total_bytes || 0) / 1073741824);
        var cells = backends.map(function(b) { return getCell(config, b); });
        var bestIdx = -1; var bestVal = -Infinity;
        cells.forEach(function(c, i) {
          if (c.type === 'data' && c.speedup > bestVal) { bestVal = c.speedup; bestIdx = i; }
        });

        return h('tr', {key: config},
          h('td', null, config, ' ', h('span', {className: 'config-hw'},
            '(' + (hw.cpu_count_logical || '?') + 'c / ' + ramGB + 'G)')),
          cells.map(function(c, i) {
            if (c.type === 'skipped') return h('td', {key: backends[i],
              style: {color: '#9CA3AF', fontStyle: 'italic'}, title: c.reason}, 'skip');
            if (c.type === 'missing') return h('td', {key: backends[i],
              style: {color: '#D1D5DB'}}, '—');
            return h('td', {key: backends[i],
              className: i === bestIdx ? 'best' : '',
              style: {background: getSpeedupColor(c.speedup), cursor: 'pointer'},
              onClick: function() { props.onCellClick({config: config, backend: backends[i]}); }},
              c.speedup.toFixed(2),
              c.correctnessFailed ? h('span', {className: 'cell-warning',
                title: c.correctnessError}, ' ⚠') : null
            );
          })
        );
      }),
      // Show message if no data at all for this filter combo
      configs.every(function(c) {
        return backends.every(function(b) { return getCell(c, b).type !== 'data'; });
      }) ? h('tr', null, h('td', {colSpan: backends.length + 1,
        style: {textAlign: 'center', color: '#9CA3AF', padding: '20px'}},
        'No data for this combination')) : null
      )
    )
  );
}
```

- [ ] **Step 4: Write CellDetailModal component**

```javascript
// append to profiling/dashboard_components.js

function CellDetailModal(props) {
  var data = props.data, f = props.filters;
  var config = props.config, backend = props.backend;

  useEffect(function() {
    function onKey(e) { if (e.key === 'Escape') props.onClose(); }
    document.addEventListener('keydown', onKey);
    return function() { document.removeEventListener('keydown', onKey); };
  }, []);

  var cfg = data.configs[config];
  var entry = cfg.timing.find(function(t) {
    return t.backend === backend && t.operation === f.operation &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var profiled = cfg.timing.find(function(t) {
    return t.backend === backend && t.operation === 'run_mlp_profiled' &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var npEntry = cfg.timing.find(function(t) {
    return t.backend === 'numpy' && t.operation === f.operation &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var speedup = entry ? entry.speedup_vs_numpy : null;
  var configs = Object.keys(data.configs);
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');

  function sel(label, key, opts, val) {
    return h('div', null,
      h('div', {className: 'filter-label'}, label),
      h('select', {className: 'filter-select', value: val,
        onChange: function(e) { props.onFilterChange(key, e.target.value); }},
        opts.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
      )
    );
  }

  return h('div', {className: 'modal-overlay', onClick: function(e) {
      if (e.target.className === 'modal-overlay') props.onClose(); }},
    h('div', {className: 'modal'},
      // Header
      h('div', {className: 'modal-header'},
        h('div', {style: {display: 'flex', alignItems: 'center', gap: '10px'}},
          h('span', {className: 'modal-title'}, backend),
          speedup != null ? h('span', {className: speedup >= 1 ? 'badge-faster' : 'badge-slower'},
            speedup.toFixed(1) + 'x ' + (speedup >= 1 ? 'faster' : 'slower')) : null
        ),
        h('button', {className: 'modal-close', onClick: props.onClose}, '✕')
      ),
      // Filter row
      h('div', {className: 'modal-filters'},
        configs.length > 1 ? sel('Config', 'modal', configs.map(function(c) { return c; }), config) : null,
        sel('Width', 'width', widths, f.width),
        sel('Depth', 'depth', depths, f.depth),
        sel('N Samples', 'nSamples', nSamples, f.nSamples)
      ),
      // Body
      h('div', {className: 'modal-body'},
        // Left: timing
        h('div', null,
          entry ? [
            h('div', {key: 'big', className: 'timing-big'},
              formatTime(entry.median_time), h('span', {className: 'timing-unit'}, '')),
            h('div', {key: 'sub', className: 'timing-sub'},
              'median of ' + (entry.times || []).length + ' runs'),
            (entry.times || []).map(function(t, i) {
              var maxT = Math.max.apply(null, entry.times);
              var pct = (t / maxT * 100).toFixed(0);
              var isMedian = Math.abs(t - entry.median_time) < 1e-9;
              return h('div', {key: i, className: 'run-bar-row'},
                h('span', {className: 'run-bar-label'}, '#' + (i + 1)),
                h('div', {className: 'run-bar-track'},
                  h('div', {className: 'run-bar-fill', style: {width: pct + '%',
                    background: isMedian ? 'var(--success)' : 'var(--chart-3)'}},
                    h('span', {className: 'run-bar-value'}, formatTime(t))
                  )
                )
              );
            }),
            (function() {
              var times = entry.times || [];
              if (times.length < 2) return null;
              var mean = times.reduce(function(a, b) { return a + b; }, 0) / times.length;
              var variance = times.reduce(function(a, t) { return a + (t - mean) * (t - mean); }, 0) / (times.length - 1);
              var std = Math.sqrt(variance);
              var cv = (std / mean * 100).toFixed(1);
              return h('div', {key: 'stats', className: 'timing-stats'},
                'σ = ' + formatTime(std) + '  CV = ' + cv + '%');
            })()
          ] : h('div', {style: {color: '#9CA3AF'}}, 'No timing data')
        ),
        // Right: breakdown
        h('div', null,
          profiled && profiled.breakdown ? [
            h('div', {key: 'label', className: 'section-header'}, 'OPERATION BREAKDOWN'),
            h('div', {key: 'bar', className: 'breakdown-bar'},
              h('span', {style: {width: profiled.breakdown.matmul_pct + '%', background: '#334155'}},
                profiled.breakdown.matmul_pct > 15 ? profiled.breakdown.matmul_pct.toFixed(0) + '%' : ''),
              h('span', {style: {width: profiled.breakdown.relu_pct + '%', background: '#F0524D'}},
                profiled.breakdown.relu_pct > 15 ? profiled.breakdown.relu_pct.toFixed(0) + '%' : ''),
              h('span', {style: {width: profiled.breakdown.overhead_pct + '%', background: '#E5E7EB'}}, '')
            ),
            h('div', {key: 'legend', className: 'breakdown-legend'},
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#334155'}}), 'MatMul'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.total_matmul)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.matmul_pct.toFixed(1) + '%'))
              ),
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#F0524D'}}), 'ReLU'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.total_relu)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.relu_pct.toFixed(1) + '%'))
              ),
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#E5E7EB'}}), 'Overhead'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.overhead)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.overhead_pct.toFixed(1) + '%'))
              )
            ),
            profiled.breakdown.matmul_per_layer ? h('div', {key: 'spark'},
              h('div', {className: 'section-header', style: {marginTop: '14px'}}, 'MATMUL PER LAYER'),
              h('div', {className: 'sparkline'},
                profiled.breakdown.matmul_per_layer.map(function(v, i) {
                  var maxV = Math.max.apply(null, profiled.breakdown.matmul_per_layer);
                  return h('div', {key: i, className: 'sparkline-bar',
                    style: {height: (v / maxV * 100) + '%'}, title: 'Layer ' + (i+1) + ': ' + formatTime(v)});
                })
              ),
              h('div', {className: 'sparkline-labels'},
                h('span', null, '1'),
                h('span', null, String(profiled.breakdown.matmul_per_layer.length))
              )
            ) : null
          ] : h('div', {style: {color: '#9CA3AF', padding: '20px 0'}},
            'No profiled data for this configuration')
        )
      ),
      // Footer: vs numpy comparison
      entry && npEntry ? h('div', {className: 'modal-footer'},
        h('span', {className: 'compare-label'}, 'numpy'),
        h('div', {className: 'compare-bar'},
          h('div', {className: 'compare-fill', style: {
            width: '100%', background: BACKEND_COLORS.numpy}},
            h('span', null, formatTime(npEntry.median_time)))
        ),
        h('span', {className: 'compare-label'}, backend),
        h('div', {className: 'compare-bar'},
          h('div', {className: 'compare-fill', style: {
            width: Math.min(entry.median_time / npEntry.median_time * 100, 100) + '%',
            background: BACKEND_COLORS[backend] || '#666'}},
            h('span', null, formatTime(entry.median_time)))
        )
      ) : null
    )
  );
}
```

- [ ] **Step 5: Write CPUScalingChart component**

```javascript
// append to profiling/dashboard_components.js

function CPUScalingChart(props) {
  var data = props.data, f = props.filters;
  var configs = Object.keys(data.configs);
  if (configs.length <= 1) return null;

  var LC = Recharts.LineChart, L = Recharts.Line, XA = Recharts.XAxis,
      YA = Recharts.YAxis, TT = Recharts.Tooltip, Lg = Recharts.Legend,
      RC = Recharts.ResponsiveContainer;

  var fs = useState('all');
  var family = fs[0]; var setFamily = fs[1];

  // Sort configs by vCPU count
  var sorted = configs.slice().sort(function(a, b) {
    return (data.configs[a].hardware.cpu_count_logical || 0) -
           (data.configs[b].hardware.cpu_count_logical || 0);
  }).filter(function(c) {
    if (family === 'all') return true;
    return c.indexOf(family) === 0;
  });

  // Build chart data: [{name, numpy, jax, ...}, ...]
  var backends = {};
  sorted.forEach(function(c) {
    data.configs[c].timing.forEach(function(t) {
      if (t.operation === f.operation && t.width === f.width &&
          t.depth === f.depth && t.n_samples === f.nSamples) {
        backends[t.backend] = true;
      }
    });
  });
  var backendList = Object.keys(backends);

  var chartData = sorted.map(function(c) {
    var hw = data.configs[c].hardware;
    var point = {name: c + ' (' + hw.cpu_count_logical + 'c)'};
    data.configs[c].timing.forEach(function(t) {
      if (t.operation === f.operation && t.width === f.width &&
          t.depth === f.depth && t.n_samples === f.nSamples) {
        point[t.backend] = t.median_time;
      }
    });
    return point;
  });

  return h('section', {className: 'scaling-section'},
    h('div', {className: 'section-header'}, 'CPU SCALING'),
    h('div', {className: 'heatmap-filters'},
      h('div', null,
        h('div', {className: 'filter-label'}, 'Config Family'),
        h('select', {className: 'filter-select', value: family,
          onChange: function(e) { setFamily(e.target.value); }},
          h('option', {value: 'all'}, 'All'),
          h('option', {value: 'compute'}, 'Compute'),
          h('option', {value: 'general'}, 'General')
        )
      )
    ),
    h('div', {className: 'scaling-chart-container'},
      h(RC, {width: '100%', height: 300},
        h(LC, {data: chartData, margin: {top: 5, right: 30, left: 20, bottom: 5}},
          h(XA, {dataKey: 'name', fontSize: 10}),
          h(YA, {fontSize: 10, label: {value: 'Time (s)', angle: -90, position: 'insideLeft'}}),
          h(TT, null),
          h(Lg, null),
          backendList.map(function(b) {
            return h(L, {key: b, type: 'monotone', dataKey: b,
              stroke: BACKEND_COLORS[b] || '#666', strokeWidth: 2, dot: {r: 4}});
          })
        )
      )
    )
  );
}
```

- [ ] **Step 6: Write DataTable component**

```javascript
// append to profiling/dashboard_components.js

function DataTable(props) {
  var data = props.data;
  var st = useState({sortKey: 'config', sortDir: 'asc', filters: {}});
  var state = st[0]; var setState = st[1];

  // Flatten all timing entries
  var rows = [];
  Object.keys(data.configs).forEach(function(config) {
    var cfg = data.configs[config];
    cfg.timing.forEach(function(t) {
      rows.push({config: config, backend: t.backend, operation: t.operation,
        width: t.width, depth: t.depth, n_samples: t.n_samples,
        median_time: t.median_time, speedup: t.speedup_vs_numpy,
        hardware: cfg.hardware, backend_versions: cfg.backend_versions});
    });
  });

  // Apply filters
  var filtered = rows.filter(function(r) {
    for (var k in state.filters) {
      if (state.filters[k] && String(r[k]) !== String(state.filters[k])) return false;
    }
    return true;
  });

  // Sort
  filtered.sort(function(a, b) {
    var av = a[state.sortKey], bv = b[state.sortKey];
    if (av < bv) return state.sortDir === 'asc' ? -1 : 1;
    if (av > bv) return state.sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  function toggleSort(key) {
    setState(function(prev) {
      return {sortKey: key, sortDir: prev.sortKey === key && prev.sortDir === 'asc' ? 'desc' : 'asc',
        filters: prev.filters};
    });
  }

  function setFilter(key, val) {
    setState(function(prev) {
      var f = {}; for (var k in prev.filters) f[k] = prev.filters[k];
      f[key] = val || '';
      return {sortKey: prev.sortKey, sortDir: prev.sortDir, filters: f};
    });
  }

  var cols = [
    {key: 'config', label: 'Config'}, {key: 'backend', label: 'Backend'},
    {key: 'operation', label: 'Operation'}, {key: 'width', label: 'Width'},
    {key: 'depth', label: 'Depth'}, {key: 'n_samples', label: 'N Samples'},
    {key: 'median_time', label: 'Median Time'}, {key: 'speedup', label: 'Speedup'}
  ];

  var expanded = useState({});
  var exp = expanded[0]; var setExp = expanded[1];

  return h('section', {className: 'datatable-section'},
    h('div', {className: 'section-header'}, 'RAW DATA'),
    h('table', {className: 'datatable'},
      h('thead', null,
        h('tr', null, cols.map(function(c) {
          var arrow = state.sortKey === c.key ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
          return h('th', {key: c.key, onClick: function() { toggleSort(c.key); }},
            c.label, h('span', {className: 'sort-arrow'}, arrow));
        })),
        h('tr', null, cols.map(function(c) {
          var vals = uniqueVals(rows, c.key);
          return h('th', {key: c.key + '-f', style: {padding: '4px'}},
            vals.length <= 20 ? h('select', {className: 'filter-select',
              style: {width: '100%', fontSize: '9px'},
              value: state.filters[c.key] || '',
              onChange: function(e) { setFilter(c.key, e.target.value); }},
              h('option', {value: ''}, 'All'),
              vals.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
            ) : null
          );
        }))
      ),
      h('tbody', null, filtered.map(function(r, i) {
        var rowKey = r.config + '-' + r.backend + '-' + r.operation + '-' + r.width + '-' + r.depth + '-' + r.n_samples;
        var isExp = exp[rowKey];
        return [
          h('tr', {key: rowKey, onClick: function() {
            setExp(function(prev) { var n = {}; for (var k in prev) n[k] = prev[k]; n[rowKey] = !prev[rowKey]; return n; });
          }, style: {cursor: 'pointer'}},
            h('td', {style: {fontFamily: 'var(--font-sans)'}}, r.config),
            h('td', null, r.backend),
            h('td', null, r.operation),
            h('td', null, r.width),
            h('td', null, r.depth),
            h('td', null, r.n_samples),
            h('td', null, formatTime(r.median_time)),
            h('td', {style: {fontWeight: r.speedup > 1.1 ? '600' : '400',
              color: r.speedup > 1 ? 'var(--success)' : 'var(--coral)'}}, r.speedup.toFixed(2) + 'x')
          ),
          isExp ? h('tr', {key: rowKey + '-detail'},
            h('td', {colSpan: 8, style: {background: 'var(--gray-50)', fontSize: '10px',
              fontFamily: 'var(--font-mono)', padding: '8px 12px'}},
              'CPU: ' + (r.hardware.cpu_count_logical || '?') +
              ' | RAM: ' + Math.round((r.hardware.ram_total_bytes || 0) / 1073741824) + 'GB' +
              ' | Platform: ' + (r.hardware.platform || '?') +
              (r.backend_versions ? ' | Versions: ' + JSON.stringify(r.backend_versions) : '')
            )
          ) : null
        ];
      }))
    )
  );
}
```

- [ ] **Step 7: Wire up ReactDOM.render**

```javascript
// append to bottom of profiling/dashboard_components.js

ReactDOM.render(h(App, null), document.getElementById('root'));
```

- [ ] **Step 8: Commit**

```bash
git add profiling/dashboard_components.js
git commit -m "feat(dashboard): add all React components (heatmap, modal, chart, table)"
```

---

## Chunk 4: HTML Assembly & End-to-End

### Task 6: HTML assembly function

**Files:**
- Modify: `profiling/generate_dashboard.py`
- Modify: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write test for HTML assembly**

```python
# append to profiling/tests/test_generate_dashboard.py

from profiling.generate_dashboard import generate_html


def test_generate_html_structure():
    html = generate_html(MULTI_CONFIG_DATA)
    assert "<!DOCTYPE html>" in html
    assert "nestim Profiling Dashboard" in html
    # Data should be embedded
    assert "window.__PROFILING_DATA__" in html
    assert "compute-small" in html
    # Should contain React components
    assert "SpeedupHeatmap" in html or "App" in html
    # Should contain CSS variables
    assert "--coral" in html


def test_generate_html_single_config():
    normalized = normalize_data(SINGLE_CONFIG_DATA)
    html = generate_html(normalized)
    assert "<!DOCTYPE html>" in html
    assert "window.__PROFILING_DATA__" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py::test_generate_html_structure -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement generate_html**

```python
# add to profiling/generate_dashboard.py

def generate_html(data):
    """Generate complete self-contained HTML dashboard."""
    # 1. Extract base CSS from network-explorer
    base_css = extract_base_css()

    # 2. Load dashboard-specific CSS
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_styles.css")
    with open(css_path) as f:
        dashboard_css = f.read()

    # 3. Fetch CDN libs
    libs = fetch_cdn_libs()

    # 4. Load React components JS
    js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_components.js")
    with open(js_path) as f:
        components_js = f.read()

    # 5. Serialize data
    data_json = json.dumps(data, default=str)

    # 6. Assemble HTML
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nestim Profiling Dashboard</title>
<script>{libs['react']}</script>
<script>{libs['react-dom']}</script>
<script>{libs['recharts']}</script>
<style>
{base_css}
{dashboard_css}
</style>
</head>
<body>
<div id="root"></div>
<script>window.__PROFILING_DATA__ = {data_json};</script>
<script>
{components_js}
</script>
</body>
</html>"""
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add profiling/generate_dashboard.py profiling/tests/test_generate_dashboard.py
git commit -m "feat(dashboard): add HTML assembly with inlined libs and CSS"
```

---

### Task 7: Main entry point and end-to-end test

**Files:**
- Modify: `profiling/generate_dashboard.py`
- Modify: `profiling/tests/test_generate_dashboard.py`

- [ ] **Step 1: Write end-to-end test**

```python
# append to profiling/tests/test_generate_dashboard.py

def test_end_to_end_generates_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fin:
        json.dump(MULTI_CONFIG_DATA, fin)
        input_path = fin.name

    output_path = input_path.replace(".json", ".html")
    try:
        from profiling.generate_dashboard import main
        main(["--input", input_path, "--output", output_path])
        assert os.path.exists(output_path)
        with open(output_path) as f:
            html = f.read()
        assert len(html) > 10000  # Should be substantial
        assert "compute-small" in html
        assert "<!DOCTYPE html>" in html
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
```

- [ ] **Step 2: Implement main function**

```python
# add to profiling/generate_dashboard.py

def main(argv=None):
    """Main entry point for dashboard generation."""
    args = parse_args(argv)
    input_path, output_path = resolve_paths(args)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    raw_data = load_data(input_path)
    data = normalize_data(raw_data)

    config_count = len(data.get("configs", {}))
    print(f"Generating dashboard for {config_count} config(s)...")

    html = generate_html(data)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest profiling/tests/test_generate_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run end-to-end with real data**

Run: `python -m profiling.generate_dashboard --run-id 2026-03-20-213311-6150c93-dirty`
Expected: Writes `profiling/results/2026-03-20-213311-6150c93-dirty/dashboard.html`

- [ ] **Step 5: Open in browser and verify**

Open the generated HTML file in a browser. Verify:
- Header shows run ID, git commit, timestamp
- Heatmap renders with colored cells
- Clicking a cell opens the modal
- Modal shows timing bars and breakdown (if profiled data exists)
- CPU scaling chart renders (if multi-config)
- Data table is sortable

- [ ] **Step 6: Commit**

```bash
git add profiling/generate_dashboard.py profiling/tests/test_generate_dashboard.py
git commit -m "feat(dashboard): add main entry point and end-to-end generation"
```

---

### Task 8: Integration and cleanup

**Files:**
- Modify: `profiling/collect_results.py` (optional: add `--dashboard` flag)

- [ ] **Step 1: Test with single-config local data**

Run: `python -m profiling.generate_dashboard --input profiling/results/2026-03-20-212951-6150c93-dirty/compute-small.json --output /tmp/local-dashboard.html`
Expected: Generates dashboard with single row heatmap, no scaling chart

- [ ] **Step 2: Verify in browser**

Open `/tmp/local-dashboard.html` in browser. Verify single-config mode works correctly.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest profiling/tests/ -v`
Expected: All tests PASS (including existing tests for other profiling modules)

- [ ] **Step 4: Final commit**

```bash
git add profiling/generate_dashboard.py profiling/dashboard_components.js profiling/dashboard_styles.css profiling/tests/test_generate_dashboard.py .gitignore
git commit -m "feat(dashboard): complete profiling dashboard generator"
```
