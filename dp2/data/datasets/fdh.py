import torch
import tops
import numpy as np
import io
import webdataset as wds
import os
from ..utils import png_decoder, mask_decoder, get_num_workers, collate_fn


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = torch.from_numpy(np.load(io.BytesIO(x))).float()
    check_outside = lambda x: (x < 0).logical_or(x > 1)
    is_outside = check_outside(keypoints[:, 0]).logical_or(
        check_outside(keypoints[:, 1])
    )
    keypoints[:, 2] = (keypoints[:, 2] > 0).logical_and(is_outside.logical_not())
    return keypoints


def vertices_decoder(x):
    vertices = torch.from_numpy(np.load(io.BytesIO(x)).astype(np.int32))
    return vertices.squeeze()[None]


def get_dataloader_fdh_wds(
        path,
        batch_size: int,
        num_workers: int,
        transform: torch.nn.Module,
        gpu_transform: torch.nn.Module,
        infinite: bool,
        shuffle: bool,
        partial_batches: bool,
        load_embedding: bool,
        sample_shuffle=10_000,
        tar_shuffle=100,
        read_condition=False,
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
        wds.handle_extension("mask.png", mask_decoder),
        wds.handle_extension("maskrcnn_mask.png", mask_decoder),
        wds.handle_extension("keypoints.npy", kp_decoder),
    ]

    rename_keys = [
        ["img", "image.png"], ["mask", "mask.png"],
        ["keypoints", "keypoints.npy"], ["maskrcnn_mask", "maskrcnn_mask.png"]
    ]
    if load_embedding:
        decoder.extend([
            wds.handle_extension("vertices.npy", vertices_decoder),
            wds.handle_extension("E_mask.png", mask_decoder)
        ])
        rename_keys.extend([
            ["vertices", "vertices.npy"],
            ["E_mask", "e_mask.png"]
        ])

    if read_condition:
        decoder.append(
            wds.handle_extension("condition.png", png_decoder)
        )
        rename_keys.append(["condition", "condition.png"])

    pipeline.extend([
        wds.tarfile_to_samples(),
        wds.decode(*decoder),
        wds.rename_keys(*rename_keys),
        wds.batched(batch_size, collation_fn=collate_fn, partial=partial_batches),
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
