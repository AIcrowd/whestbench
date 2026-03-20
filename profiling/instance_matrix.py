"""Fargate task configuration matrix for cloud profiling.

Defines CPU/memory combinations that simulate compute-optimized (c-series)
and general-purpose (m-series) EC2 instance characteristics on Fargate.

To customize, add or remove entries from INSTANCE_MATRIX. Each entry needs:
- name: unique identifier (used as S3 key prefix and in reports)
- cpu: Fargate CPU units (1024 = 1 vCPU)
- memory: Fargate memory in MiB
- label: human-readable description
"""

from typing import Dict, List, Optional, Any

Config = Dict[str, Any]

INSTANCE_MATRIX: List[Config] = [
    # Compute-optimized (c-series equivalents) — low memory-to-CPU ratio
    {"name": "compute-small",   "cpu": 1024,  "memory": 2048,  "label": "1 vCPU / 2 GB"},
    {"name": "compute-medium",  "cpu": 2048,  "memory": 4096,  "label": "2 vCPU / 4 GB"},
    {"name": "compute-large",   "cpu": 4096,  "memory": 8192,  "label": "4 vCPU / 8 GB"},
    {"name": "compute-xlarge",  "cpu": 8192,  "memory": 16384, "label": "8 vCPU / 16 GB"},
    {"name": "compute-2xlarge", "cpu": 16384, "memory": 32768, "label": "16 vCPU / 32 GB"},
    # General-purpose (m-series equivalents) — higher memory-to-CPU ratio
    {"name": "general-small",   "cpu": 1024,  "memory": 4096,  "label": "1 vCPU / 4 GB"},
    {"name": "general-medium",  "cpu": 2048,  "memory": 8192,  "label": "2 vCPU / 8 GB"},
    {"name": "general-large",   "cpu": 4096,  "memory": 16384, "label": "4 vCPU / 16 GB"},
    {"name": "general-xlarge",  "cpu": 8192,  "memory": 32768, "label": "8 vCPU / 32 GB"},
]


def get_configs(names: Optional[List[str]] = None) -> List[Config]:
    """Return configs filtered by name. Returns all if names is None."""
    if names is None:
        return list(INSTANCE_MATRIX)
    known = {c["name"] for c in INSTANCE_MATRIX}
    unknown = set(names) - known
    if unknown:
        raise ValueError(
            f"Unknown config(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(known))}"
        )
    return [c for c in INSTANCE_MATRIX if c["name"] in names]
