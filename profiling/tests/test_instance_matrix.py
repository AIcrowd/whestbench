"""Tests for the Fargate instance matrix."""

from profiling.instance_matrix import INSTANCE_MATRIX, get_configs


def test_matrix_has_compute_and_general_configs():
    names = [c["name"] for c in INSTANCE_MATRIX]
    compute = [n for n in names if n.startswith("compute-")]
    general = [n for n in names if n.startswith("general-")]
    assert len(compute) >= 3
    assert len(general) >= 3


def test_all_configs_have_required_keys():
    required = {"name", "cpu", "memory", "label"}
    for config in INSTANCE_MATRIX:
        assert required.issubset(config.keys()), f"Missing keys in {config['name']}"


def test_cpu_memory_are_valid_fargate_combos():
    """Fargate valid CPU values: 256, 512, 1024, 2048, 4096, 8192, 16384."""
    valid_cpu = {256, 512, 1024, 2048, 4096, 8192, 16384}
    for config in INSTANCE_MATRIX:
        assert config["cpu"] in valid_cpu, f"Invalid CPU {config['cpu']} for {config['name']}"
        assert config["memory"] >= config["cpu"], f"Memory must be >= CPU for {config['name']}"


def test_get_configs_returns_all_by_default():
    configs = get_configs()
    assert len(configs) == len(INSTANCE_MATRIX)


def test_get_configs_filters_by_name():
    configs = get_configs(names=["compute-small", "general-small"])
    assert len(configs) == 2
    assert {c["name"] for c in configs} == {"compute-small", "general-small"}


def test_get_configs_raises_on_unknown_name():
    try:
        get_configs(names=["nonexistent"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
