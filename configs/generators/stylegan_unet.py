from dp2.generator.stylegan_unet import StyleGANUnet
from tops.config import LazyCall as L

generator = L(StyleGANUnet)(
    imsize="${data.imsize}",
    im_channels="${data.im_channels}",
    min_fmap_resolution=8,
    cnum=64,
    max_cnum_mul=8,
    n_middle_blocks=0,
    z_channels=512,
    mask_output=True,
    conv_clamp=256,
    input_cse=True,
    scale_grad=True,
    cse_nc="${data.cse_nc}",
    w_dim=512,
    n_keypoints="${data.n_keypoints}",
    input_keypoints=False,
    input_keypoint_indices=[],
    fix_errors=True
)