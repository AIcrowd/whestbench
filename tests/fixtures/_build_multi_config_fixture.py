"""Regenerate the frozen two-config (config-per-split) fixture.

Run from the repo root:
    uv run python tests/fixtures/_build_multi_config_fixture.py

Produces tests/fixtures/multi_config_v3.0/ matching the published evals shape:
  config `default` -> split `public`,  config `holdout` -> split `holdout`.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    from whestbench.dataset import create_dataset
    from whestbench.dataset_io import combine_split_datasets

    here = Path(__file__).parent
    work = here / "_multi_config_work"
    dest = here / "multi_config_v3.0"
    for path in (work, dest):
        if path.exists():
            shutil.rmtree(path)

    def bake(name: str, split: str, config: str, seed: int) -> Path:
        out = work / name
        create_dataset(
            n_mlps=2,
            n_samples=100,
            width=8,
            depth=2,
            mlp_seeds=[seed * 1000 + i for i in range(2)],
            output_path=out,
            split=split,
            config=config,
        )
        return out

    pub = bake("pub", "public", "default", 42)
    hold = bake("hold", "holdout", "holdout", 99)
    combine_split_datasets([pub, hold], output_dir=dest, write_prepared_arrow=False)
    shutil.rmtree(work)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
