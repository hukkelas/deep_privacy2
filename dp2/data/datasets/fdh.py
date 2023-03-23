import torch
import tops
import numpy as np
import io
import webdataset as wds
import os
import json
from pathlib import Path
from ..utils import png_decoder, mask_decoder, get_num_workers, collate_fn


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = torch.from_numpy(np.load(io.BytesIO(x))).float()
    def check_outside(x): return (x < 0).logical_or(x > 1)
    is_outside = check_outside(keypoints[:, 0]).logical_or(
        check_outside(keypoints[:, 1])
    )
    keypoints[:, 2] = (keypoints[:, 2] > 0).logical_and(is_outside.logical_not())
    return keypoints


def vertices_decoder(x):
    vertices = torch.from_numpy(np.load(io.BytesIO(x)).astype(np.int32))
    return vertices.squeeze()[None]


class InsertNewKeypoints:

    def __init__(self, keypoints_path: Path) -> None:
        with open(keypoints_path, "r") as fp:
            self.keypoints = json.load(fp)

    def __call__(self, sample):
        key = sample["__key__"]
        keypoints = torch.tensor(self.keypoints[key], dtype=torch.float32)
        def check_outside(x): return (x < 0).logical_or(x > 1)
        is_outside = check_outside(keypoints[:, 0]).logical_or(
            check_outside(keypoints[:, 1])
        )
        keypoints[:, 2] = (keypoints[:, 2] > 0).logical_and(is_outside.logical_not())

        sample["keypoints.npy"] = keypoints
        return sample


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
        load_new_keypoints=False,
        keypoints_split=None,
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
        ["keypoints", "keypoints.npy"], ["maskrcnn_mask", "maskrcnn_mask.png"],
        ["__key__", "__key__"]
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

    ])
    if load_new_keypoints:
        assert keypoints_split in ["train", "val"]
        keypoint_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/1eb88522-8b91-49c7-b56a-ed98a9c7888cef9c0429-a385-4248-abe3-8682de26d041f268aed1-7c88-4677-baad-7623c2ee330f"
        file_name = "fdh_keypoints_val-050133b34d.json"
        if keypoints_split == "train":
            keypoint_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/3e828b1c-d6c0-4622-90bc-1b2cce48ccfff14ab45d-0a5c-431d-be13-7e60580765bd7938601c-e72e-41d9-8836-fffc49e76f58"
            file_name = "fdh_keypoints_train-2cff11f69a.json"
        # Set check_hash=True if you suspect download is incorrect.
        filepath = tops.download_file(keypoint_url, file_name=file_name, check_hash=False)
        pipeline.append(
            wds.map(InsertNewKeypoints(filepath))
        )
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
