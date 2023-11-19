from .im18 import (
    G_optim, D_optim, loss_fnc, common, EMA, train,
    generator, discriminator
)
from .....datasets.fdh_resampled_wds.fdh72 import data

generator.update(
    dim_mults=[1, 1, 1, 1],
)

train.update(
    generator_init_cfg="configs/fdh/triagan/ablations/PG/im36.py",
)
train.max_images_to_train = int(50e6)