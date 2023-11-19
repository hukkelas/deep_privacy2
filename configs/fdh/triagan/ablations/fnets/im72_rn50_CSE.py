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
from .....discriminators.projected_discriminator import RN50_CSE

RN50_CSE.interp_size = (288, 160)
discriminator.update(
    backbones=[RN50_CSE],
)
