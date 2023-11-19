import os
from pathlib import Path
from tops.config import LazyCall as L
import torch
import functools
from dp2.data.datasets.fdf128_wds import get_dataloader_fdf_wds
from dp2.data.utils import get_fdf_flipmap
from dp2.data.transforms.transforms import (
    Normalize,
    ToFloat,
    CreateCondition,
    RandomHorizontalFlip,
)
from ..utils import train_eval_fn
from dp2.metrics.torch_metrics import compute_metrics_iteratively
from dp2.metrics.fid_clip import compute_fid_clip
from dp2.metrics.ppl import calculate_ppl


def final_eval_fn(*args, **kwargs):
    result = compute_metrics_iteratively(*args, **kwargs)
    result2 = calculate_ppl(*args, **kwargs, upsample_size=(128, 128))
    result3 = compute_fid_clip(*args, **kwargs)
    assert all(key not in result for key in result2)
    result.update(result2)
    result.update(result3)
    return result


def get_cache_directory(imsize, subset):
    return Path(metrics_cache, f"{subset}{imsize[0]}")

dataset_base_dir = (
    os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
)
metrics_cache = (
    os.environ["FBA_METRICS_CACHE"] if "FBA_METRICS_CACHE" in os.environ else ".cache"
)
data_dir = Path(dataset_base_dir, "fdf_resampled")
data = dict(
    imsize=(128, 128),
    im_channels=3,
    cse_nc=0,
    n_keypoints=7,
    train=dict(
        loader=L(get_dataloader_fdf_wds)(
            path=data_dir.joinpath("train", "128", "out-{000000..000473}.tar"),
            batch_size="${train.batch_size}",
            num_workers=4,
            transform=L(torch.nn.Sequential)(
                L(RandomHorizontalFlip)(p=0.5, flip_map=get_fdf_flipmap()),
            ),
            gpu_transform=L(torch.nn.Sequential)(
                L(ToFloat)(norm=False, keys=["img", "mask"]),
                L(Normalize)(mean=[0.5*255, 0.5*255, 0.5*255], std=[0.5*255, 0.5*255, 0.5*255], inplace=True),
                L(CreateCondition)(),
            ),
            infinite=True,
            shuffle=True,
            partial_batches=False,
        )
    ),
    val=dict(
        loader=L(get_dataloader_fdf_wds)(
            path=data_dir.joinpath("val", "128", "out-{000000..000016}.tar"),
            batch_size="${train.batch_size}",
            num_workers=2,
            transform=None,
            gpu_transform="${data.train.loader.gpu_transform}",
            infinite=False,
            shuffle=False,
            partial_batches=True,
        )
    ),
    # Training evaluation might do optimizations to reduce compute overhead. E.g. compute with AMP.
    train_evaluation_fn=L(functools.partial)(
        train_eval_fn, cache_directory=L(get_cache_directory)(imsize="${data.imsize}", subset="fdf"),
        data_len=50_000),
    evaluation_fn=L(functools.partial)(
        final_eval_fn, cache_directory=L(get_cache_directory)(imsize="${data.imsize}", subset="fdf_eval"), 
        data_len=50_000)
)
