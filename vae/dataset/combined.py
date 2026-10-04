from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Dict, Iterable, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .dl3dv import DL3DVSuperpointChunkDatasetV2C15
from .re10k import RE10KSuperpointChunkDatasetV2C15
from .scannet import ScanNetSuperpointChunkDatasetV2C15


DATASET_REGISTRY = {
    "scannet": ScanNetSuperpointChunkDatasetV2C15,
    "scannet_chunkv2_c15": ScanNetSuperpointChunkDatasetV2C15,
    "dl3dv": DL3DVSuperpointChunkDatasetV2C15,
    "dl3dv_chunkv2_c15": DL3DVSuperpointChunkDatasetV2C15,
    "re10k": RE10KSuperpointChunkDatasetV2C15,
    "re10k_chunkv2_c15": RE10KSuperpointChunkDatasetV2C15,
}


class ResizedDataset(Dataset):
    """Epoch-shuffled virtual length wrapper, matching GLD's `ratio @ dataset` idea."""

    def __init__(self, dataset: Dataset, new_size: int, seed_offset: int = 777) -> None:
        if int(new_size) <= 0:
            raise ValueError(f"new_size must be positive, got {new_size}")
        self.dataset = dataset
        self.new_size = int(new_size)
        self.seed_offset = int(seed_offset)
        self._idxs_mapping: Optional[np.ndarray] = None
        self.set_epoch(0)

    def __len__(self) -> int:
        if len(self.dataset) == 0:
            return 0
        return self.new_size

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if len(self.dataset) == 0:
            self._idxs_mapping = np.array([], dtype=np.int64)
            return
        rng = np.random.default_rng(seed=int(epoch) + self.seed_offset)
        perm = rng.permutation(len(self.dataset))
        repeats = 1 + (self.new_size - 1) // len(self.dataset)
        self._idxs_mapping = np.concatenate([perm] * repeats)[: self.new_size].astype(np.int64)

    def __getitem__(self, idx: int):
        if self._idxs_mapping is None:
            raise RuntimeError("ResizedDataset.set_epoch() must be called before sampling.")
        return self.dataset[int(self._idxs_mapping[int(idx)])]


class ConcatDatasetWithEpoch(Dataset):
    """Concat wrapper that propagates set_epoch to children."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        self.datasets = [d for d in datasets if len(d) > 0]
        if not self.datasets:
            raise ValueError("No non-empty datasets to concatenate.")
        self._cum_sizes = np.cumsum([len(d) for d in self.datasets])

    def __len__(self) -> int:
        return int(self._cum_sizes[-1])

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
        self._cum_sizes = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        dataset_idx = int(np.searchsorted(self._cum_sizes, idx, side="right"))
        prev = int(self._cum_sizes[dataset_idx - 1]) if dataset_idx > 0 else 0
        return self.datasets[dataset_idx][idx - prev]


class CombinedSuperpointChunkDatasetV2C15(Dataset):
    """Multi-dataset C15 chunks with GLD-style ratio resize + concat.

    Args:
        datasets: Mapping from dataset key to kwargs for that dataset.
        mix_ratio: Virtual training length per dataset. If omitted and
            equal_ratio=False, each dataset keeps its natural length.
        equal_ratio: Resize all training datasets to the largest child length.
        val_datasets: Optional validation dataset keys. Defaults to all datasets.
        val_mix_ratio: Optional validation virtual lengths. By default validation
            is plain concat without ratio resizing.
    """

    def __init__(
        self,
        split: str = "train",
        datasets: Optional[Dict[str, Dict]] = None,
        mix_ratio: Optional[Dict[str, int]] = None,
        equal_ratio: bool = False,
        val_datasets: Optional[List[str]] = None,
        val_mix_ratio: Optional[Dict[str, int]] = None,
        **shared_kwargs,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("CombinedSuperpointChunkDatasetV2C15 requires non-empty `datasets` config.")
        self.split = split
        self.dataset_specs = datasets
        self.mix_ratio = mix_ratio or {}
        self.equal_ratio = bool(equal_ratio)
        self.val_mix_ratio = val_mix_ratio

        names = list(datasets.keys())
        if split != "train" and val_datasets is not None:
            names = [n for n in val_datasets if n in datasets]
        if not names:
            raise ValueError(f"No dataset names selected for split={split}.")

        children = []
        for name in names:
            key = _normalize_name(name)
            cls = DATASET_REGISTRY.get(key)
            if cls is None:
                raise ValueError(f"Unknown combined dataset child: {name}")
            child_kwargs = _merge_kwargs(shared_kwargs, datasets[name])
            child_kwargs = _filter_kwargs_for_class(cls, child_kwargs)
            child = cls(split=split, **child_kwargs)
            if len(child) == 0:
                continue
            children.append((name, child))
        if not children:
            raise ValueError(f"No non-empty child datasets for split={split}.")

        if split == "train":
            self.dataset = self._build_train_dataset(children)
        else:
            self.dataset = self._build_val_dataset(children)

    def _build_train_dataset(self, children: List[tuple[str, Dataset]]) -> Dataset:
        if self.equal_ratio:
            target = max(len(child) for _, child in children)
            return ConcatDatasetWithEpoch(ResizedDataset(child, target) for _, child in children)

        resized = []
        for name, child in children:
            target = self.mix_ratio.get(name, self.mix_ratio.get(_normalize_name(name), len(child)))
            resized.append(ResizedDataset(child, int(target)))
        return ConcatDatasetWithEpoch(resized)

    def _build_val_dataset(self, children: List[tuple[str, Dataset]]) -> Dataset:
        if self.val_mix_ratio:
            resized = []
            for name, child in children:
                target = self.val_mix_ratio.get(name, self.val_mix_ratio.get(_normalize_name(name), len(child)))
                resized.append(ResizedDataset(child, int(target)))
            return ConcatDatasetWithEpoch(resized)
        return ConcatDatasetWithEpoch(child for _, child in children)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def _normalize_name(name: str) -> str:
    return str(name).lower()


def _merge_kwargs(shared_kwargs: Dict, child_kwargs: Dict) -> Dict:
    out = deepcopy(shared_kwargs)
    out.pop("datasets", None)
    out.pop("mix_ratio", None)
    out.pop("equal_ratio", None)
    out.pop("val_datasets", None)
    out.pop("val_mix_ratio", None)
    out.update(deepcopy(child_kwargs or {}))
    return out


def _filter_kwargs_for_class(cls, kwargs: Dict) -> Dict:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
