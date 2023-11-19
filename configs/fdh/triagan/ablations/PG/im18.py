from ..base import generator, discriminator, D_optim, G_optim, common, EMA, train, loss_fnc
from .....datasets.fdh_resampled_wds.fdh18 import data

generator.update(
    dim_mults=[1, 1],
    layer_scale=True,
)
train.max_images_to_train = int(76e6)