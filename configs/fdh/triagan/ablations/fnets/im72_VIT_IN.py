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
from .....discriminators.projected_discriminator import deit_base_distilled_patch16_224

deit_base_distilled_patch16_224.interp_size = (224, 224)
discriminator.update(
    backbones=[deit_base_distilled_patch16_224],
)
