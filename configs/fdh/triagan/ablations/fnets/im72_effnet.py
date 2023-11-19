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
from .....discriminators.projected_discriminator import (
    tf_efficientnet_lite0,
)

tf_efficientnet_lite0.interp_size = (288, 160)
discriminator.update(
    backbones=[tf_efficientnet_lite0],
)
