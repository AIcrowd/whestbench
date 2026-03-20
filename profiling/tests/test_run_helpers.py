"""Tests for run_benchmarks helper functions."""

import json
import re
from unittest.mock import patch

from profiling.run_helpers import generate_run_id, load_infra_config


def test_run_id_format_matches_pattern():
    """Run ID should be YYYY-MM-DD-HHMMSS-<7char-hash>."""
    with patch("profiling.run_helpers._git_short_hash", return_value="abc1234"):
        with patch("profiling.run_helpers._git_is_dirty", return_value=False):
            run_id = generate_run_id()
    assert re.match(r"\d{4}-\d{2}-\d{2}-\d{6}-[a-f0-9]{7}$", run_id)


def test_run_id_dirty_suffix():
    with patch("profiling.run_helpers._git_short_hash", return_value="abc1234"):
        with patch("profiling.run_helpers._git_is_dirty", return_value=True):
            run_id = generate_run_id()
    assert run_id.endswith("-dirty")


def test_run_id_override():
    run_id = generate_run_id(override="my-custom-id")
    assert run_id == "my-custom-id"


def test_load_infra_config(tmp_path):
    config = {"region": "us-east-1", "s3_bucket": "test-bucket"}
    config_path = tmp_path / ".infra-config.json"
    config_path.write_text(json.dumps(config))
    loaded = load_infra_config(str(config_path))
    assert loaded["region"] == "us-east-1"
    assert loaded["s3_bucket"] == "test-bucket"


def test_load_infra_config_missing_file():
    try:
        load_infra_config("/nonexistent/.infra-config.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
