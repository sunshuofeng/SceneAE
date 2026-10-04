from mmengine.registry import Registry
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .combined import CombinedSuperpointChunkDatasetV2C15
from .dl3dv import DL3DVSuperpointChunkDatasetV2C15
from .ffg_cached import CombinedFFGCachedDataset, FFGCachedChunkDataset
from .re10k import RE10KSuperpointChunkDatasetV2C15
from .sampler import CustomDistributedSampler
from .scannet import ScanNetSuperpointChunkDatasetV2C15

OPENOCC_DATASET = Registry("openocc_dataset")
OPENOCC_DATAWRAPPER = Registry("openocc_datawrapper")

_DATASETS = {
    "ffg_cached": CombinedFFGCachedDataset,
    "scannet": ScanNetSuperpointChunkDatasetV2C15,
    "scannet_chunkv2_c15": ScanNetSuperpointChunkDatasetV2C15,
    "dl3dv": DL3DVSuperpointChunkDatasetV2C15,
    "dl3dv_chunkv2_c15": DL3DVSuperpointChunkDatasetV2C15,
    "re10k": RE10KSuperpointChunkDatasetV2C15,
    "re10k_chunkv2_c15": RE10KSuperpointChunkDatasetV2C15,
    "combined": CombinedSuperpointChunkDatasetV2C15,
}


def get_dataloader(
    batch_size=4,
    dataset_name="ffg_cached",
    dataset_kwargs=None,
    dist=False,
    iter_resume=False,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    train_sampler_config=dict(shuffle=True, drop_last=True),
    val_sampler_config=dict(shuffle=False, drop_last=False),
    collate_fn=None,
):
    dataset_kwargs = dataset_kwargs or {}
    key = str(dataset_name).lower()
    if key not in _DATASETS:
        raise KeyError(f"Unknown dataset_name={dataset_name!r}. Known: {sorted(_DATASETS)}")
    cls = _DATASETS[key]
    train_dataset = cls(split="train", **dataset_kwargs)
    val_dataset = cls(split="test", **dataset_kwargs)
    train_sampler = val_sampler = None
    if dist:
        sampler_cls = CustomDistributedSampler if iter_resume else DistributedSampler
        train_sampler = sampler_cls(train_dataset, **train_sampler_config)
        val_sampler = DistributedSampler(val_dataset, **val_sampler_config)
    num_workers = int(num_workers)
    persistent_workers = bool(persistent_workers) and num_workers > 0
    loader_kw = dict(pin_memory=True, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=persistent_workers)
    if num_workers > 0:
        loader_kw["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False if dist else True, sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, **loader_kw)
    return train_loader, val_loader
