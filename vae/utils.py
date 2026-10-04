
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    

    

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
   
    torch.cuda.manual_seed_all(seed)
  
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
      
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


from collections import OrderedDict


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if not tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {('module.' + k): v
                for k, v in state_dict.items()})
    return state_dict

def revise_ckpt_1(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v
                for k, v in state_dict.items()})
    return state_dict
        
def revise_ckpt_2(state_dict):
    param_names = list(state_dict.keys())
    for param_name in param_names:
        if 'img_neck.lateral_convs' in param_name or 'img_neck.fpn_convs' in param_name:
            del state_dict[param_name]
    return state_dict

def collate_points(batch, max_points=None):
    """
    Concatenate variable-length point clouds and keep offsets.
    For keys in ["norm_pw", "norm_points_world", "points_world"].
    """
    out = {}
    for k in batch[0]:
        
        vals = [b[k] for b in batch]
        if k in ["norm_pw", "norm_points_world", "points_world"]:
            offsets = []
            cat = []
            total = 0
            
            for v in vals:
                v = torch.as_tensor(v)
                
                cat.append(v)
                total += v.shape[0]
                offsets.append(total)
            out[k] = torch.cat(cat, dim=0)

            out[f"{k}_offsets"] = torch.tensor(offsets, dtype=torch.long)
        else:
            out[k] = torch.utils.data._utils.collate.default_collate(vals)
    return out