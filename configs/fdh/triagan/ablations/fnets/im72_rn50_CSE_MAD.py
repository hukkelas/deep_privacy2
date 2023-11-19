from .im72_rn50_CSE import (
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

loss_fnc.loss_type = "masked-hinge"
