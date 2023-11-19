from .L_im36 import (
    G_optim, D_optim, loss_fnc, common, EMA, train,
    generator, discriminator, ckpt_mapper
)
from ...datasets.fdh_resampled_wds.fdh72 import data
generator.update(
    dim_mults=[1, 1, 1, 1],
)

train.update(
    generator_init_cfg="configs/fdh/triagan/L_im36.py",
)
train.max_images_to_train = int(160e6)

common.model_md5sum = "291b9374a8f004fb9a2cfce59249be58"
common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/57f4101a-e1b7-414d-b616-ce4d71586fc7379379b4-c16e-41ba-aa3f-cd0f5af1ae94ecbce694-137f-4c80-8f7c-0b7ed6ace39f"