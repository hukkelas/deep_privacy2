from tops.config import LazyCall as L
from dp2.discriminator import SG2Discriminator
import torch
from dp2.loss import StyleGAN2Loss


discriminator = L(SG2Discriminator)(
    imsize="${data.imsize}",
    im_channels="${data.im_channels}",
    min_fmap_resolution=4,
    max_cnum_mul=8,
    cnum=80,
    input_condition=True,
    conv_clamp=256,
    input_cse=False,
    cse_nc="${data.cse_nc}",
    fix_residual=False,
)


loss_fnc = L(StyleGAN2Loss)(
    lazy_regularization=True,
    lazy_reg_interval=16,
    r1_opts=dict(lambd=5, mask_out=False, mask_out_scale=False),
    EP_lambd=0.001,
    pl_reg_opts=dict(weight=0, batch_shrink=2,start_nimg=int(1e6), pl_decay=0.01)
)

def build_D_optim(type, lr, betas, lazy_regularization, lazy_reg_interval, **kwargs):
    if lazy_regularization:
        # From Analyzing and improving the image quality of stylegan, CVPR 2020
        c = lazy_reg_interval / (lazy_reg_interval + 1)
        betas = [beta ** c for beta in betas]
        lr *= c
        print(f"Lazy regularization on. Setting lr to: {lr}, betas to: {betas}")
    return type(lr=lr, betas=betas, **kwargs)


D_optim = L(build_D_optim)(
    type=torch.optim.Adam, lr=0.001, betas=(0.0, 0.99),
    lazy_regularization="${loss_fnc.lazy_regularization}",
    lazy_reg_interval="${loss_fnc.lazy_reg_interval}")
G_optim = L(torch.optim.Adam)(lr=0.001, betas=(0.0, 0.99))
