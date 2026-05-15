#!/usr/bin/env python3
"""
Run all visualization scripts for the MSA project.
Each script is run as a subprocess for isolation.
Output: output/figures/fig*.png
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "fig01_cumulative_excess",
    "fig02_drawdown_comparison",
    "fig03_block_excess_hist",
    "fig04_knn_neighbor_time",
    "fig05_pca_knn",
    "fig06_fdr_threshold",
    "fig07_dsr_distribution",
]


def main():
    viz_dir = Path(__file__).resolve().parent
    project_root = viz_dir.parent.parent

    for name in SCRIPTS:
        path = viz_dir / f"{name}.py"
        if not path.exists():
            print(f"[SKIP] {name}.py not found")
            continue
        print(f"\n{'=' * 55}")
        print(f"  Running {name}.py...")
        print(f"{'=' * 55}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(project_root),
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] exit code {result.returncode}")


if __name__ == "__main__":
    main()
