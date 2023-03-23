import os
from pathlib import Path
from tops.config import LazyCall as L
import torch
import functools
from dp2.data.datasets.fdf import FDF256Dataset
from dp2.data.build import get_dataloader
from dp2.data.transforms.transforms import Normalize, Resize, ToFloat, CreateCondition, RandomHorizontalFlip
from .utils import final_eval_fn, train_eval_fn


dataset_base_dir = os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
metrics_cache = os.environ["FBA_METRICS_CACHE"] if "FBA_METRICS_CACHE" in os.environ else ".cache"
data_dir = Path(dataset_base_dir, "fdf256")
data = dict(
    imsize=(256, 256),
    im_channels=3,
    semantic_nc=None,
    cse_nc=None,
    n_keypoints=None,
    train=dict(
        dataset=L(FDF256Dataset)(dirpath=data_dir.joinpath("train"), transform=None, load_keypoints=False),
        loader=L(get_dataloader)(
            shuffle=True, num_workers=3, drop_last=True, prefetch_factor=2,
            batch_size="${train.batch_size}",
            dataset="${..dataset}",
            infinite=True,
            gpu_transform=L(torch.nn.Sequential)(*[
                L(ToFloat)(),
                L(RandomHorizontalFlip)(p=0.5),
                L(Resize)(size="${data.imsize}"),
                L(Normalize)(mean=[.5, .5, .5], std=[.5, .5, .5], inplace=True),
                L(CreateCondition)(),
            ])
        )
    ),
    val=dict(
        dataset=L(FDF256Dataset)(dirpath=data_dir.joinpath("val"), transform=None, load_keypoints=False),
        loader=L(get_dataloader)(
            shuffle=False, num_workers=3, drop_last=False, prefetch_factor=2,
            batch_size="${train.batch_size}",
            dataset="${..dataset}",
            infinite=False,
            gpu_transform=L(torch.nn.Sequential)(*[
                L(ToFloat)(),
                L(Resize)(size="${data.imsize}"),
                L(Normalize)(mean=[.5, .5, .5], std=[.5, .5, .5], inplace=True),
                L(CreateCondition)(),
            ])
        )
    ),
    # Training evaluation might do optimizations to reduce compute overhead. E.g. compute with AMP.
    train_evaluation_fn=functools.partial(train_eval_fn, cache_directory=Path(metrics_cache, "fdf_val_train")),
    evaluation_fn=functools.partial(final_eval_fn, cache_directory=Path(metrics_cache, "fdf_val"))
)