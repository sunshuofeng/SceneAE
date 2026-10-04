#!/usr/bin/env python3
"""Cache 2-view stage-1 xyz. See cache_stage1_xyz.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv[1:1] = [
    "--stage1_config", "configs/stage1_2view.py",
    "--dataset_config", "configs/stage2_2view.py",
    "--two_view",
]
from tools.cache_stage1_xyz import main

if __name__ == "__main__":
    main()
