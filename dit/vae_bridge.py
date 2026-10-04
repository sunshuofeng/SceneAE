"""Import the frozen VAE helpers without putting vae/ on the DiT import path permanently."""

from __future__ import annotations

import sys
from pathlib import Path

_VAE = Path(__file__).resolve().parents[1] / "vae"
if str(_VAE) not in sys.path:
    sys.path.insert(0, str(_VAE))

from cache_latent import load_json_config, load_model  # noqa: E402

__all__ = ["load_json_config", "load_model"]
