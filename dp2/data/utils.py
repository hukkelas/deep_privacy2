import torch
from PIL import Image
import numpy as np
import multiprocessing
import io
from tops import logger
from torch.utils.data._utils.collate import default_collate

try:
    import pyspng

    PYSPNG_IMPORTED = True
except ImportError:
    PYSPNG_IMPORTED = False
    print("Could not load pyspng. Defaulting to pillow image backend.")
    from PIL import Image


def get_fdf_keypoints():
    return get_coco_keypoints()[:7]


def get_fdf_flipmap():
    keypoints = get_fdf_keypoints()
    keypoint_flip_map = {
        "left_eye": "right_eye",
        "left_ear": "right_ear",
        "left_shoulder": "right_shoulder",
    }
    for key, value in list(keypoint_flip_map.items()):
        keypoint_flip_map[value] = key
    keypoint_flip_map["nose"] = "nose"
    keypoint_flip_map_idx = []
    for source in keypoints:
        keypoint_flip_map_idx.append(keypoints.index(keypoint_flip_map[source]))
    return keypoint_flip_map_idx


def get_coco_keypoints():
    return [
        "nose",
        "left_eye",
        "right_eye",  # 2
        "left_ear",
        "right_ear",  # 4
        "left_shoulder",
        "right_shoulder",  # 6
        "left_elbow",
        "right_elbow",  # 8
        "left_wrist",
        "right_wrist",  # 10
        "left_hip",
        "right_hip",  # 12
        "left_knee",
        "right_knee",  # 14
        "left_ankle",
        "right_ankle",  # 16
    ]


def get_coco_flipmap():
    keypoints = get_coco_keypoints()
    keypoint_flip_map = {
        "left_eye": "right_eye",
        "left_ear": "right_ear",
        "left_shoulder": "right_shoulder",
        "left_elbow": "right_elbow",
        "left_wrist": "right_wrist",
        "left_hip": "right_hip",
        "left_knee": "right_knee",
        "left_ankle": "right_ankle",
    }
    for key, value in list(keypoint_flip_map.items()):
        keypoint_flip_map[value] = key
    keypoint_flip_map["nose"] = "nose"
    keypoint_flip_map_idx = []
    for source in keypoints:
        keypoint_flip_map_idx.append(keypoints.index(keypoint_flip_map[source]))
    return keypoint_flip_map_idx


def mask_decoder(x):
    mask = torch.from_numpy(np.array(Image.open(io.BytesIO(x)))).squeeze()[None]
    mask = mask > 0  # This fixes bug causing  maskf.loat().max() == 255.
    return mask


def png_decoder(x):
    if PYSPNG_IMPORTED:
        return torch.from_numpy(np.rollaxis(pyspng.load(x), 2))
    with Image.open(io.BytesIO(x)) as im:
        im = torch.from_numpy(np.rollaxis(np.array(im.convert("RGB")), 2))
    return im


def jpg_decoder(x):
    with Image.open(io.BytesIO(x)) as im:
        im = torch.from_numpy(np.rollaxis(np.array(im.convert("RGB")), 2))
    return im


def get_num_workers(num_workers: int):
    n_cpus = multiprocessing.cpu_count()
    if num_workers > n_cpus:
        logger.warn(f"Setting the number of workers to match cpu count: {n_cpus}")
        return n_cpus
    return num_workers


def collate_fn(batch):
    elem = batch[0]
    ignore_keys = set(["embed_map", "vertx2cat"])
    batch_ = {
        key: default_collate([d[key] for d in batch])
        for key in elem
        if key not in ignore_keys
    }
    if "embed_map" in elem:
        batch_["embed_map"] = elem["embed_map"]
    if "vertx2cat" in elem:
        batch_["vertx2cat"] = elem["vertx2cat"]
    return batch_
