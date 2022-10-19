import torch 
import tops
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    elem = batch[0]
    ignore_keys = set(["embed_map", "vertx2cat"])
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem if key not in ignore_keys}
    if "embed_map" in elem:
        batch_["embed_map"] = elem["embed_map"]
    if "vertx2cat" in elem:
        batch_["vertx2cat"] = elem["vertx2cat"]
    return batch_


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