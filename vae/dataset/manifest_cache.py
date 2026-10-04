from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = 2


def default_manifest_path(root: Path, split: str, num_views: int, dataset_name: str) -> Path:
    root = Path(root)
    return root / f".{dataset_name}_index_{split}_views{int(num_views)}.json"


def load_manifest(path: Optional[Path], root: Path, split: str, num_views: int, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    meta = payload.get("meta", {})
    if int(meta.get("version", -1)) != MANIFEST_VERSION:
        return None
    if str(meta.get("dataset_name", "")) != str(dataset_name):
        return None
    if str(meta.get("root", "")) != str(Path(root).resolve()):
        return None
    if str(meta.get("split", "")) != str(split):
        return None
    if int(meta.get("num_views", -1)) != int(num_views):
        return None

    items = payload.get("items", None)
    if not isinstance(items, list):
        return None
    return items


def save_manifest(
    path: Optional[Path],
    root: Path,
    split: str,
    num_views: int,
    dataset_name: str,
    items: List[Dict[str, Any]],
) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "version": MANIFEST_VERSION,
            "dataset_name": str(dataset_name),
            "root": str(Path(root).resolve()),
            "split": str(split),
            "num_views": int(num_views),
            "num_items": int(len(items)),
        },
        "items": items,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(path)
    except OSError:
        return


def path_to_manifest(path: Path) -> str:
    return str(Path(path).resolve())


def path_from_manifest(path: str) -> Path:
    return Path(path)
