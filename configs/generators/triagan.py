from dp2.generator.triagan import TriaGAN
from functools import partial
from tops.config import LazyCall as L
from dp2.generator.base import SG2StyleNet

generator = L(TriaGAN)(
    im_channels="${data.im_channels}",
    imsize="${data.imsize}",
    dim=512,
    dim_mults=[1, 1],
    num_resnet_blocks=1,
    n_middle_blocks=2,
    z_channels=64,
    w_dim=512,
    norm_enc=True,
    use_maskrcnn_mask=True,
    input_keypoints=False,
    input_keypoint_indices=[],
    input_joint=True,
    n_keypoints="${data.n_keypoints}",
    norm_dec=True,
    layer_scale=True,
    use_noise=True,
    style_net=L(SG2StyleNet)(
        z_dim="${generator.z_channels}", w_dim="${generator.w_dim}"
    ),
)


def ckpt_map_G(state: dict):
    new_state = dict()
    for key, item in state.items():
        new_key = key
        for i in range(6):
            if key == f"decoder.to_rgb.{i}.weight":
                new_key = "decoder.to_rgb.weight"
            elif key == f"decoder.to_rgb.{i}.bias":
                new_key = "decoder.to_rgb.bias"
        new_state[new_key] = item
    return new_state


ckpt_mapper = L(partial)(ckpt_map_G)
