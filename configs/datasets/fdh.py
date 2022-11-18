import os
from pathlib import Path
from tops.config import LazyCall as L
import torch
import functools
from dp2.data.datasets.fdh import get_dataloader_fdh_wds
from dp2.data.utils import get_coco_flipmap
from dp2.data.transforms.transforms import (
    Normalize,
    ToFloat,
    CreateCondition,
    RandomHorizontalFlip,
    CreateEmbedding,
)
from dp2.metrics.torch_metrics import compute_metrics_iteratively
from dp2.metrics.fid_clip import compute_fid_clip
from .utils import final_eval_fn


def train_eval_fn(*args, **kwargs):
    result = compute_metrics_iteratively(*args, **kwargs)
    result2 = compute_fid_clip(*args, **kwargs)
    assert all(key not in result for key in result2)
    result.update(result2)
    return result


dataset_base_dir = (
    os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
)
metrics_cache = (
    os.environ["FBA_METRICS_CACHE"] if "FBA_METRICS_CACHE" in os.environ else ".cache"
)
data_dir = Path(dataset_base_dir, "fdh")
data = dict(
    imsize=(288, 160),
    im_channels=3,
    cse_nc=16,
    n_keypoints=17,
    train=dict(
        loader=L(get_dataloader_fdh_wds)(
            path=data_dir.joinpath("train", "out-{000000..001421}.tar"),
            batch_size="${train.batch_size}",
            num_workers=6,
            transform=L(torch.nn.Sequential)(
                L(RandomHorizontalFlip)(p=0.5, flip_map=get_coco_flipmap()),
            ),
            gpu_transform=L(torch.nn.Sequential)(
                L(ToFloat)(norm=False, keys=["img", "mask", "E_mask", "maskrcnn_mask"]),
                L(CreateEmbedding)(embed_path=data_dir.joinpath("embed_map.torch")),
                L(Normalize)(mean=[0.5*255, 0.5*255, 0.5*255], std=[0.5*255, 0.5*255, 0.5*255], inplace=True),
                L(CreateCondition)(),
            ),
            infinite=True,
            shuffle=True,
            partial_batches=False,
            load_embedding=True,
        )
    ),
    val=dict(
        loader=L(get_dataloader_fdh_wds)(
            path=data_dir.joinpath("val", "out-{000000..000023}.tar"),
            batch_size="${train.batch_size}",
            num_workers=6,
            transform=None,
            gpu_transform=L(torch.nn.Sequential)(
                L(ToFloat)(keys=["img", "mask", "E_mask", "maskrcnn_mask"], norm=False),
                L(CreateEmbedding)(embed_path=data_dir.joinpath("embed_map.torch")),
                L(Normalize)(mean=[0.5*255, 0.5*255, 0.5*255], std=[0.5*255, 0.5*255, 0.5*255], inplace=True),
                L(CreateCondition)(),
            ),
            infinite=False,
            shuffle=False,
            partial_batches=True,
            load_embedding=True,
        )
    ),
    # Training evaluation might do optimizations to reduce compute overhead. E.g. compute with AMP.
    train_evaluation_fn=functools.partial(
        train_eval_fn,
        cache_directory=Path(metrics_cache, "fdh_val"),
        data_len=int(30e3),
    ),
    evaluation_fn=functools.partial(
        final_eval_fn,
        cache_directory=Path(metrics_cache, "fdh_final_val"),
        data_len=int(30e3),
    ),
)
