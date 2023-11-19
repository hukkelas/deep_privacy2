from functools import partial
import torch
from dp2.discriminator.projected_gan import ProjectedDiscriminator
from dp2.discriminator.projected_gan.feature_nets import (
    _make_efficientnet,
    _make_resnet50_clip,
    _make_resnet50_cse,
    _make_resnet50,
)
from dp2.discriminator.projected_gan.vit import _make_vit_timm
from dp2.discriminator.projected_gan.vit_openclip import OpenCLIPViT

from dp2.loss.projected_gan_loss import ProjectedGANLoss, blur
from tops.config import LazyCall as L

imagenet_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
inception_norm = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
clip_norm = dict(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

VIT_PATCH_MAE = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="vit_base_patch16_224",
        weight_path="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
    ),
    interp_size=None,
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)
VIT_PATCH_MAE_LARGE = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="vit_large_patch16_224",
        weight_path="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
    ),
    interp_size=None,
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)
VIT_PATCH_MAE_HUGE = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="vit_huge_patch14_224",
        weight_path="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
    ),
    interp_size=None,
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)
VIT_DINO_S = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="vit_small_patch16_224",
        weight_path="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    ),
    interp_size=(224, 224),
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)
VIT_DINO_B = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="vit_base_patch16_224",
        weight_path="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    ),
    interp_size=(224, 224),
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)

tf_efficientnet_lite0 = dict(
    backbone_cfg=L(_make_efficientnet)(model_type="tf_efficientnet_lite0"),
    interp_size=None,
    input_BGR=False,
    jit_script=True,
    **inception_norm
)
RN50_CSE = dict(
    backbone_cfg=L(_make_resnet50_cse)(
        cfg_url="https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml"
    ),
    interp_size=None,
    mean=[0.485, 0.456, 0.406],
    std=[1 / 255, 1 / 255, 1 / 255],
    input_BGR=True,
    jit_script=False
)
RN50_CLIP = dict(
    backbone_cfg=L(_make_resnet50_clip)(),
    interp_size=None,
    input_BGR=False,
    jit_script=False,
    **clip_norm
)
RN50_IN = dict(
    backbone_cfg=L(_make_resnet50)(),
    interp_size=None,
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)

deit_base_distilled_patch16_224 = dict(
    backbone_cfg=L(_make_vit_timm)(
        model_name="deit_base_distilled_patch16_224", weight_path=None
    ),
    interp_size=(224, 224),
    input_BGR=False,
    jit_script=False,
    **imagenet_norm
)

openclip_vit_b_32 = dict(
    backbone_cfg=L(OpenCLIPViT)(
        model_name="ViT-B-32",
        pretrained_dataset="laion2b_s34b_b79k",
    ),
    input_BGR=False,
    jit_script=False,
    interp_size=(224, 224),
    **clip_norm # Not sure if norm is correct for L-14 / B-32
)
openclip_vit_b_16 = dict(
    backbone_cfg=L(OpenCLIPViT)(
        model_name="ViT-B-16",
        pretrained_dataset="openai",
    ),
    input_BGR=False,
    jit_script=False,
    interp_size=(224, 224),
    **clip_norm 
)
openclip_vit_l_14 = dict(
    backbone_cfg=L(OpenCLIPViT)(
        model_name="ViT-L-14",
        pretrained_dataset="laion2b_s32b_b82k",
    ),
    input_BGR=False,
    jit_script=False,
    interp_size=(224, 224),
    **clip_norm # Not sure if norm is correct for L-14 / B-32
)


discriminator = L(ProjectedDiscriminator)(
    num_discs=4,
    patch=True,
    diffaug_policy=["color", "cutout"],
    backbones=[VIT_PATCH_MAE],
    last_ksize=3,
)

loss_fnc = L(ProjectedGANLoss)(
    aug_fade_kimg=int(400e3),
    blur_init_sigma=4,
)

D_optim = L(torch.optim.Adam)(lr=0.002, betas=(0.0, 0.99))
G_optim = L(torch.optim.Adam)(lr=0.002, betas=(0.0, 0.99))
