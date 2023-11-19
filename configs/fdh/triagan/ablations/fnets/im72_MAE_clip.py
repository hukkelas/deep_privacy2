from ..base import (
    G_optim,
    D_optim,
    loss_fnc,
    common,
    EMA,
    train,
    data,
    generator,
    discriminator,
    ckpt_mapper,
)
from .....discriminators.projected_discriminator import VIT_PATCH_MAE_LARGE, RN50_CLIP

VIT_PATCH_MAE_LARGE.interp_size = (224, 224)
RN50_CLIP.interp_size = (288, 160)
discriminator.update(
    backbones=[VIT_PATCH_MAE_LARGE, RN50_CLIP],
)
