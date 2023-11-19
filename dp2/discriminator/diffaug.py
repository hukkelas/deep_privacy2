# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
# Code modified from: https://github.com/autonomousvision/projected_gan/blob/main/pg_modules/diffaug.py
import torch
from typing import List, Optional, Tuple
from torch.nn.functional import interpolate
from torchvision.transforms.functional import affine


def diff_augment(img, condition=None, policies: Optional[List[str]] = None):
    if policies is None:
        return img, condition
    for policy in policies:
        fns = AUGMENT_FNS[policy]
        for fn in fns:
            img, condition = fn(img, condition)
    return img, condition


def rand_brightness(img: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
    aug = torch.rand(img.size(0), 1, 1, 1, dtype=img.dtype, device=img.device) - 0.5
    img = img + aug
    if condition is not None:
        condition = condition + aug
    return img, condition


def rand_saturation(img: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
    aug = torch.rand(img.size(0), 1, 1, 1, dtype=img.dtype, device=img.device) * 2
    x_mean = img.mean(dim=1, keepdim=True)
    img = (img - x_mean) * aug + x_mean
    if condition is not None:
        if condition.shape != x_mean.shape:
            x_mean = interpolate(x_mean, size=condition.shape[-2:], antialias=True, mode="bilinear")
        condition = (condition - x_mean) * aug + x_mean
    return img, condition


def rand_contrast(img: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
    x_mean = img.mean(dim=[1, 2, 3], keepdim=True)
    aug = torch.rand(img.size(0), 1, 1, 1, dtype=img.dtype, device=img.device) + 0.5
    img = (img - x_mean) * aug + x_mean
    if condition is not None:
        condition = (condition - x_mean) * aug + x_mean
    return img, condition


def rand_cutout(img: torch.Tensor, condition: torch.Tensor, ratio=.2) -> Tuple[torch.Tensor]:
    cutout_size = int(img.size(2) * ratio + 0.5), int(img.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, img.size(2) + (1 - cutout_size[0] % 2), size=[img.size(0), 1, 1], device=img.device)
    offset_y = torch.randint(0, img.size(3) + (1 - cutout_size[1] % 2), size=[img.size(0), 1, 1], device=img.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(img.size(0), dtype=torch.long, device=img.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=img.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=img.device),
        indexing="ij")
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=img.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=img.size(3) - 1)
    mask = torch.ones(img.size(0), img.size(2), img.size(3), dtype=img.dtype, device=img.device)
    mask[grid_batch, grid_x, grid_y] = 0
    img = img * mask.unsqueeze(1)
    if condition is not None:
        if condition.shape != mask.shape:
            mask = mask.unsqueeze(1)
            mask = interpolate(mask, condition.shape[-2:], mode="nearest")
        condition = condition * mask
    return img, condition


def rand_translation(img: torch.Tensor, condition: torch.Tensor, ratio=.2) -> Tuple[torch.Tensor]:
    xy_shift = (torch.rand(size=(2,)) * 2 - 1).tolist()
    xy_shift[0] *= img.shape[-1] * ratio
    xy_shift[1] *= img.shape[-2] * ratio
    img = affine(img, angle=0, translate=xy_shift, scale=1, shear=[0, 0])
    if condition is not None:
        condition = affine(condition, angle=0, translate=xy_shift, scale=1, shear=[0, 0])
    return img, condition


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'cutout': [rand_cutout],
    "translate": [rand_translation],
}
