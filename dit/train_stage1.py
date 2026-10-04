"""GLD/DDP training entrypoint for v18_v3 stage-1 position DiT.

The flow/loss logic is identical to v16_v3 stage-1 (flow matching interpolant, x0
parameterization, x0-space masked MSE, optional first-frame partial-noise source
prior), so it is reused verbatim. This entrypoint only forwards the extra ray-map
diagnostics that the v16_v3 logger does not know about. v18_v3 vs v18_v2: backbone
shrunk to 768d x 8 layers via config (num_groups=4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import stage1_loop as v16_v3
import train_common as gld

RAYMAP_LOG_KEYS = (
    "raymap_out_norm",
    "raymap_q_valid_frac",
    "raymap_known_depth_frac",
)


def model_step(model, batch, cfg, **kwargs):
    loss, logs = v16_v3.model_step(model, batch, cfg, **kwargs)
    core_model = model.module if hasattr(model, "module") else model
    pop = getattr(core_model, "pop_diagnostic", None)
    if pop is not None:
        for key in RAYMAP_LOG_KEYS:
            value = pop(key)
            if value is not None:
                logs[key] = value
    return loss, logs


def main() -> None:
    args = gld.parse_args()
    if "--config" not in sys.argv:
        args.config = Path("configs/stage1.py")
    gld.model_step = model_step
    gld.train(args)


if __name__ == "__main__":
    main()
