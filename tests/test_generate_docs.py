import argparse
import inspect

import scripts.generate_docs as gd


def test_symbol_kind_classifies_class_function_value():
    assert gd.symbol_kind(int) == "class"
    assert gd.symbol_kind(inspect.getdoc) == "function"
    assert gd.symbol_kind(42) == "value"


def test_signature_str_falls_back_to_name_for_unsupported():
    def f(a, b=1):
        return a + b

    assert gd.signature_str("f", f) == "f(a, b=1)"
    assert gd.signature_str("X", 42) == "X"


def test_public_symbols_returns_all_whestbench_exports():
    names = [name for name, _ in gd.public_symbols()]
    assert "BaseEstimator" in names
    assert "MLP" in names
    assert names == sorted(set(names), key=names.index)


def test_render_api_page_contains_signature_and_docstring():
    def add(a, b):
        """Add two numbers."""
        return a + b

    page = gd.render_api_page("add", add)
    assert "title: add" in page
    assert "```python\nadd(a, b)\n```" in page
    assert "Add two numbers." in page


def test_write_api_writes_index_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "API_DIR", tmp_path / "api")
    written = gd.write_api()
    assert (tmp_path / "api" / "index.mdx").exists()
    assert (tmp_path / "api" / "meta.json").exists()
    assert any(p.name == "mlp.mdx" for p in written)


def _toy_parser():
    p = argparse.ArgumentParser(prog="whest")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a thing.")
    run.add_argument("--width", type=int, default=256, help="Layer width.")
    hidden = sub.add_parser("secret", help=argparse.SUPPRESS)
    hidden.add_argument("--x")
    return p


def test_iter_subcommands_skips_suppressed():
    names = [name for name, _sub, _help in gd.iter_subcommands(_toy_parser())]
    assert "run" in names
    assert "secret" not in names


def test_render_cli_page_lists_options():
    p = _toy_parser()
    run = dict((n, s) for n, s, _h in gd.iter_subcommands(p))["run"]
    page = gd.render_cli_page("run", run, "Run a thing.")
    assert "whest run" in page
    assert "--width" in page
    assert "Layer width." in page
    assert "256" in page


def test_write_cli_writes_index_and_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "CLI_DIR", tmp_path / "cli")
    written = gd.write_cli()
    assert (tmp_path / "cli" / "index.mdx").exists()
    assert (tmp_path / "cli" / "meta.json").exists()
    names = {p.stem for p in written}
    assert "run" in names and "validate" in names


def test_verify_reports_missing_docstrings(monkeypatch):
    class NoDoc:
        pass

    monkeypatch.setattr(gd, "public_symbols", lambda: [("NoDoc", NoDoc)])
    monkeypatch.setattr(gd, "cli_parser", _toy_parser)
    problems = gd.verify()
    assert any("NoDoc" in p for p in problems)


def test_verify_passes_for_documented(monkeypatch):
    def documented():
        """It is documented."""

    monkeypatch.setattr(gd, "public_symbols", lambda: [("documented", documented)])
    monkeypatch.setattr(gd, "cli_parser", _toy_parser)
    assert gd.verify() == []


def test_mdx_escapes_special_chars():
    assert gd._mdx("a < b") == "a &lt; b"
    assert gd._mdx("use {x}") == "use \\{x}"
    assert gd._mdx("x | y") == "x \\| y"


def test_render_cli_page_escapes_help_in_table_and_prose():
    p = argparse.ArgumentParser(prog="whest")
    sub = p.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("demo", help="demo cmd")
    cmd.add_argument("--n", help="ints < 2**63 in {a,b} or x|y")
    demo = dict((n, s) for n, s, _h in gd.iter_subcommands(p))["demo"]
    page = gd.render_cli_page("demo", demo, "demo cmd")
    assert "ints &lt; 2**63 in \\{a,b} or x\\|y" in page
