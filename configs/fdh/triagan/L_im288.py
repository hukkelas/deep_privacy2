from .L_im144 import (
    G_optim, D_optim, loss_fnc, common, EMA, train,
    generator, discriminator, ckpt_mapper
)
from ...datasets.fdh_no_embeddings import data
from ...discriminators.projected_discriminator import (
    VIT_PATCH_MAE_LARGE,
    RN50_CLIP
)
from functools import partial
from tops.config import LazyCall as L
generator.update(
    dim=128,
    dim_mults=[1, 2, 4, 4, 4, 4],
)
train.update(
    generator_init_cfg="configs/fdh/triagan/L_im144.py",
)
train.max_images_to_train = int(70e6)
train.batch_size = 128
VIT_PATCH_MAE_LARGE.interp_size = (224, 224)
RN50_CLIP.interp_size = None
discriminator.update(
    backbones=[VIT_PATCH_MAE_LARGE, RN50_CLIP],
)
common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/e62a306a-5295-47dd-8a0e-4f7c05b9ecd44c9fbc76-5ce2-4dd0-9454-04776bef2745659fbec7-fa0a-485b-8501-c94a2439b601"
common.model_md5sum = "b649410e831b45f7800a89c738b411ce"