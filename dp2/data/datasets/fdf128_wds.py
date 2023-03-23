import torch
import tops
import numpy as np
import io
import webdataset as wds
import os
from ..utils import png_decoder, get_num_workers, collate_fn


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = torch.from_numpy(np.load(io.BytesIO(x))).float().view(7, 2).clamp(0, 1)
    keypoints = torch.cat((keypoints, torch.ones((7, 1))), dim=-1)
    return keypoints


def bbox_decoder(x):
    return torch.from_numpy(np.load(io.BytesIO(x))).float().view(4)


class BBoxToMask:

    def __call__(self, sample):
        imsize = sample["image.png"].shape[-1]
        bbox = sample["bounding_box.npy"] * imsize
        x0, y0, x1, y1 = np.round(bbox).astype(np.int64)
        mask = torch.ones((1, imsize, imsize), dtype=torch.bool)
        mask[:, y0:y1, x0:x1] = 0
        sample["mask"] = mask
        return sample


def get_dataloader_fdf_wds(
        path,
        batch_size: int,
        num_workers: int,
        transform: torch.nn.Module,
        gpu_transform: torch.nn.Module,
        infinite: bool,
        shuffle: bool,
        partial_batches: bool,
        sample_shuffle=10_000,
        tar_shuffle=100,
        channels_last=False,
    ):
    # Need to set this for split_by_node to work.
    os.environ["RANK"] = str(tops.rank())
    os.environ["WORLD_SIZE"] = str(tops.world_size())
    if infinite:
        pipeline = [wds.ResampledShards(str(path))]
    else:
        pipeline = [wds.SimpleShardList(str(path))]
    if shuffle:
        pipeline.append(wds.shuffle(tar_shuffle))
    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
    ])
    if shuffle:
        pipeline.append(wds.shuffle(sample_shuffle))

    decoder = [
        wds.handle_extension("image.png", png_decoder),
        wds.handle_extension("keypoints.npy", kp_decoder),
    ]

    rename_keys = [
        ["img", "image.png"],
        ["keypoints", "keypoints.npy"],
        ["__key__", "__key__"],
        ["mask", "mask"]
    ]

    pipeline.extend([
        wds.tarfile_to_samples(),
        wds.decode(*decoder),
    ])
    pipeline.append(wds.map(BBoxToMask()))
    pipeline.extend([
        wds.batched(batch_size, collation_fn=collate_fn, partial=partial_batches),
        wds.rename_keys(*rename_keys),
    ])

    if transform is not None:
        pipeline.append(wds.map(transform))
    pipeline = wds.DataPipeline(*pipeline)
    if infinite:
        pipeline = pipeline.repeat(nepochs=1000000)

    loader = wds.WebLoader(
        pipeline, batch_size=None, shuffle=False,
        num_workers=get_num_workers(num_workers),
        persistent_workers=True,
    )
    loader = tops.DataPrefetcher(loader, gpu_transform, channels_last=channels_last, to_float=False)
    return loader
