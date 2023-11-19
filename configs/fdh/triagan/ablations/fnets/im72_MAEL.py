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
from .....discriminators.projected_discriminator import VIT_PATCH_MAE_LARGE

VIT_PATCH_MAE_LARGE.interp_size = (224, 224)
discriminator.update(
    backbones=[VIT_PATCH_MAE_LARGE],
)
