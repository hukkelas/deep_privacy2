import torch
import tops
from .utils import collate_fn


def get_dataloader(
        dataset, gpu_transform: torch.nn.Module,
        num_workers,
        batch_size,
        infinite: bool,
        drop_last: bool,
        prefetch_factor: int,
        shuffle,
        channels_last=False
    ):
    sampler = None
    dl_kwargs = dict(
        pin_memory=True,
    )
    if infinite:
        sampler = tops.InfiniteSampler(
            dataset, rank=tops.rank(),
            num_replicas=tops.world_size(),
            shuffle=shuffle
        )
    elif tops.world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=shuffle, num_replicas=tops.world_size(), rank=tops.rank())
        dl_kwargs["drop_last"] = drop_last
    else:
        dl_kwargs["shuffle"] = shuffle
        dl_kwargs["drop_last"] = drop_last
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        **dl_kwargs
    )
    dataloader = tops.DataPrefetcher(dataloader, gpu_transform, channels_last=channels_last)
    return dataloader
