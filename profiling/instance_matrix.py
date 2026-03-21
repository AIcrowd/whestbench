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
    # Each CPU tier gets max Fargate memory so memory is never the bottleneck.
    # We're measuring compute scaling, not memory pressure.
    # Fargate max memory per tier: 1vCPU→8GB, 2→16GB, 4→30GB, 8→60GB, 16→120GB.
    {"name": "compute-1vcpu",   "cpu": 1024,  "memory": 8192,   "label": "1 vCPU / 8 GB"},
    {"name": "compute-2vcpu",   "cpu": 2048,  "memory": 16384,  "label": "2 vCPU / 16 GB"},
    {"name": "compute-4vcpu",   "cpu": 4096,  "memory": 30720,  "label": "4 vCPU / 30 GB"},
    {"name": "compute-8vcpu",   "cpu": 8192,  "memory": 61440,  "label": "8 vCPU / 60 GB"},
    {"name": "compute-16vcpu",  "cpu": 16384, "memory": 122880, "label": "16 vCPU / 120 GB"},
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
