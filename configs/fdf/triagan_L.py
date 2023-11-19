from ..discriminators.projected_discriminator import (
    G_optim,
    D_optim,
    loss_fnc,
    discriminator,
    VIT_PATCH_MAE_LARGE,
    RN50_CLIP
)
from ..defaults import common, EMA, train
from ..datasets.fdf128_wds.fdf128 import data
from ..generators.triagan import generator, ckpt_mapper

train.update(
    batch_size=128,
    ims_per_log=1024 * 20,
    ims_per_val=int(2.4e6),
    max_images_to_train=int(77e6),
    broadcast_buffers=True,
    fp16_ddp_accumulate=True,
)
loss_fnc.aug_fade_kimg = int(4e6)

generator.update(
    dim_mults=[1, 2, 2, 2, 2],
    dim=256,
    imsize="${data.imsize}",
    use_maskrcnn_mask=False,
    input_keypoint_indices=list(range(3)),
    input_keypoints=True,
    use_noise=True,
    layer_scale=False,
    input_joint=False,
    num_resnet_blocks=2
)

VIT_PATCH_MAE_LARGE.interp_size = (224, 224)
RN50_CLIP.interp_size = (224, 224)
discriminator.update(
    backbones=[RN50_CLIP, VIT_PATCH_MAE_LARGE],
)
common.model_md5sum = "7b94366d9d2e1e9a6491e065491fda87"
common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/88e98c73-e890-489d-a3eb-cf6fbf1728f531b0047f-ddf3-4188-8028-2b55c580ac6da54d9760-e395-4f2a-8890-743f3562e1b3"