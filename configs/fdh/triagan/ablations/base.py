from ..L_im18 import (
    generator,
    discriminator,
    D_optim,
    G_optim,
    common,
    EMA,
    train,
    loss_fnc,
    ckpt_mapper,
)
from ....datasets.fdh_resampled_wds.fdh72 import data


train.update(
    batch_size=512,
    max_images_to_train=int(50e6),
)

generator.update(
    dim_mults=[1, 1, 1, 1],
    dim=512,
    num_resnet_blocks=1,
    layer_scale=False,
    use_noise=False,
)
loss_fnc.loss_type = "hinge"
