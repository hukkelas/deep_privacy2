from .L_im72 import (
    G_optim, D_optim, loss_fnc, common, EMA, train,
    generator, discriminator, ckpt_mapper
)
from ...datasets.fdh_resampled_wds.fdh144 import data
generator.update(
    dim=256,
    dim_mults=[1, 2, 2, 2,2],
)

train.update(
    generator_init_cfg="configs/fdh/triagan/L_im72.py",
)
train.max_images_to_train = int(110e6)
train.batch_size = 512