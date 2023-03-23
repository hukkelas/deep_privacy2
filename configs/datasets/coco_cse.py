import os
from pathlib import Path
from tops.config import LazyCall as L
import torch
import functools
from dp2.data.datasets.coco_cse import CocoCSE
from dp2.data.build import get_dataloader
from dp2.data.transforms.transforms import CreateEmbedding, Normalize, Resize, ToFloat, CreateCondition, RandomHorizontalFlip
from dp2.data.transforms.stylegan2_transform import StyleGANAugmentPipe
from dp2.metrics.torch_metrics import compute_metrics_iteratively
from .utils import final_eval_fn


dataset_base_dir = os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
metrics_cache = os.environ["FBA_METRICS_CACHE"] if "FBA_METRICS_CACHE" in os.environ else ".cache"
data_dir = Path(dataset_base_dir, "coco_cse")
data = dict(
    imsize=(288, 160),
    im_channels=3,
    semantic_nc=26,
    cse_nc=16,
    train=dict(
        dataset=L(CocoCSE)(data_dir.joinpath("train"), transform=None, normalize_E=False),
        loader=L(get_dataloader)(
            shuffle=True, num_workers=6, drop_last=True, prefetch_factor=2,
            batch_size="${train.batch_size}",
            dataset="${..dataset}",
            infinite=True,
            gpu_transform=L(torch.nn.Sequential)(*[
                L(ToFloat)(),
                L(StyleGANAugmentPipe)(
                    rotate=0.5, rotate_max=.05,
                    xint=.5, xint_max=0.05,
                    scale=.5, scale_std=.05,
                    aniso=0.5, aniso_std=.05,
                    xfrac=.5, xfrac_std=.05,
                    brightness=.5, brightness_std=.05,
                    contrast=.5, contrast_std=.1,
                    hue=.5, hue_max=.05,
                    saturation=.5, saturation_std=.5,
                    imgfilter=.5, imgfilter_std=.1),
                L(RandomHorizontalFlip)(p=0.5),
                L(CreateEmbedding)(),
                L(Resize)(size="${data.imsize}"),
                L(Normalize)(mean=[.5, .5, .5], std=[.5, .5, .5], inplace=True),
                L(CreateCondition)(),
            ])
        )
    ),
    val=dict(
        dataset=L(CocoCSE)(data_dir.joinpath("val"), transform=None, normalize_E=False),
        loader=L(get_dataloader)(
            shuffle=False, num_workers=6, drop_last=True, prefetch_factor=2,
            batch_size="${train.batch_size}",
            dataset="${..dataset}",
            infinite=False,
            gpu_transform=L(torch.nn.Sequential)(*[
                L(ToFloat)(),
                L(CreateEmbedding)(),
                L(Resize)(size="${data.imsize}"),
                L(Normalize)(mean=[.5, .5, .5], std=[.5, .5, .5], inplace=True),
                L(CreateCondition)(),
            ])
        )
    ),
    # Training evaluation might do optimizations to reduce compute overhead. E.g. compute with AMP.
    train_evaluation_fn=functools.partial(compute_metrics_iteratively, cache_directory=Path(metrics_cache, "coco_cse_val"), include_two_fake=False),
    evaluation_fn=functools.partial(final_eval_fn, cache_directory=Path(metrics_cache, "coco_cse_val_final"), include_two_fake=True)
)
