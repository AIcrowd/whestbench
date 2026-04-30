"""Generate self-contained HTML profiling dashboard."""

import argparse
import json
import os
import sys
import urllib.request

CDN_LIBS = {
    "react": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "prop-types": "https://unpkg.com/prop-types@15/prop-types.min.js",
    "recharts": "https://unpkg.com/recharts@2/umd/Recharts.js",
}


def load_data(path):
    """Load profiling JSON from file path."""
    with open(path) as f:
        return json.load(f)


def normalize_data(data):
    """Normalize single-config or multi-config data into multi-config format."""
    if "configs" in data:
        return data
    hostname = data.get("hardware", {}).get("hostname", "local")
    config_name = hostname if hostname != "local" else "local"
    return {
        "run_id": f"local-{config_name}",
        "git_commit": "",
        "git_dirty": False,
        "collected_at": "",
        "configs": {config_name: data},
    }


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


def extract_base_css():
    """Extract base CSS variables and resets from whestbench-explorer App.css."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(repo_root, "tools", "whestbench-explorer", "src", "App.css")
    with open(css_path) as f:
        full_css = f.read()

    # Extract everything before the App Shell section
    marker = "/* ──────────── App Shell ──────────── */"
    idx = full_css.find(marker)
    if idx == -1:
        base = full_css[:2000]
    else:
        base = full_css[:idx]

    # Also extract .app-header styles
    header_start = full_css.find(".app-header {")
    if header_start != -1:
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
                    header_section = full_css[header_start : pos + 1]
                    next_chunk = full_css[pos + 1 : pos + 50].strip()
                    if next_chunk.startswith(".app-header h1"):
                        header_start_h1 = full_css.find(".app-header h1", pos)
                        pos2 = full_css.find("}", header_start_h1)
                        header_section = full_css[header_start : pos2 + 1]
                    break
            pos += 1
        base += "\n" + header_section

    return base


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


def extract_backend_sources():
    """Read simulator backend source files and return a dict of {name: source_code}."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_dir = os.path.join(repo_root, "src", "whestbench")
    backends = {
        "numpy": "simulation_numpy.py",
        "scipy": "simulation_scipy.py",
        "numba": "simulation_numba.py",
        "cython": "simulation_cython.py",
        "jax": "simulation_jax.py",
        "pytorch": "simulation_pytorch.py",
    }
    sources = {}
    for name, filename in backends.items():
        path = os.path.join(sim_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                sources[name] = f.read()
    return sources


def generate_html(data):
    """Generate complete self-contained HTML dashboard."""
    base_css = extract_base_css()

    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_styles.css")
    with open(css_path) as f:
        dashboard_css = f.read()

    libs = fetch_cdn_libs()

    js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_components.js")
    with open(js_path) as f:
        components_js = f.read()

    data_json = json.dumps(data, default=str)
    backend_sources = extract_backend_sources()
    sources_json = json.dumps(backend_sources)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>flopscope Profiling Dashboard</title>
<script>{libs["react"]}</script>
<script>{libs["react-dom"]}</script>
<script>{libs["prop-types"]}</script>
<script>{libs["recharts"]}</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<style>
{base_css}
{dashboard_css}
</style>
</head>
<body>
<div id="root"></div>
<script>window.__PROFILING_DATA__ = {data_json};</script>
<script>window.__BACKEND_SOURCES__ = {sources_json};</script>
<script>
{components_js}
</script>
</body>
</html>"""


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
