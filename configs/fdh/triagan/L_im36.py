from .L_im18 import (
    G_optim, D_optim, loss_fnc, common, EMA, train,
    generator, discriminator, ckpt_mapper
)
from ...datasets.fdh_resampled_wds.fdh36 import data
generator.update(
    dim_mults=[1, 1, 1],
)

train.update(
    generator_init_cfg="configs/fdh/triagan/L_im18.py",
)
train.max_images_to_train = int(200e6)