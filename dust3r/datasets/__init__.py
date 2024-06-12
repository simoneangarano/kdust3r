# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import numpy as np
import random

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa: F401
from .co3d import Co3d  # noqa: F401
from .scannetpp_old import ScanNet  # noqa: F401
from .scannetpp import ScanNetpp  # noqa: F401
from .dl3dv import DL3DV  # noqa: F401
from .megadepth import MegaDepth  # noqa: F401
from .dtu import DTU  # noqa: F401
from .blendedmvs import BlendedMVS  # noqa: F401
from .scannet import Scannet  # noqa: F401

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(777)


    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g
    )

    return data_loader
