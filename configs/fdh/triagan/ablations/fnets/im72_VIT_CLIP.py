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
from .....discriminators.projected_discriminator import openclip_vit_b_16

openclip_vit_b_16.interp_size = (224, 224)
discriminator.update(
    backbones=[openclip_vit_b_16],
)
