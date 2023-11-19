from ...discriminators.projected_discriminator import (
    G_optim,
    D_optim,
    loss_fnc,
    discriminator,
    VIT_PATCH_MAE_LARGE,
    RN50_CLIP
)
from ...defaults import common, EMA, train
from dp2.data.utils import get_coco_keypoints
from ...generators.triagan import generator, ckpt_mapper
from ...datasets.fdh_resampled_wds.fdh18 import data

train.update(
    batch_size=1024,
    ims_per_log=1024 * 20,
    ims_per_val=int(10e6),
    max_images_to_train=int(300e6),
    broadcast_buffers=True,
    fp16_ddp_accumulate=True,
)
loss_fnc.aug_fade_kimg = int(4e6)

generator.update(
    input_keypoint_indices=list(range(len(get_coco_keypoints()))),
    input_keypoints=True,
    num_resnet_blocks=2,
)

loss_fnc.loss_type = "masked-hinge"
VIT_PATCH_MAE_LARGE.interp_size = (224, 224)
RN50_CLIP.interp_size = (288, 160)
discriminator.update(
    backbones=[VIT_PATCH_MAE_LARGE, RN50_CLIP],
)
