"""Generate self-contained HTML profiling dashboard."""
import argparse
import json
import os
import sys
import urllib.request


CDN_LIBS = {
    "react": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
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
    """Extract base CSS variables and resets from network-explorer App.css."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(repo_root, "tools", "network-explorer", "src", "App.css")
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
                    header_section = full_css[header_start:pos + 1]
                    next_chunk = full_css[pos + 1:pos + 50].strip()
                    if next_chunk.startswith(".app-header h1"):
                        header_start_h1 = full_css.find(".app-header h1", pos)
                        pos2 = full_css.find("}", header_start_h1)
                        header_section = full_css[header_start:pos2 + 1]
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
