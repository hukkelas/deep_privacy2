from functools import partial
from typing import Iterable, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from .base import BaseStyleGAN
from sg3_torch_utils.ops import bias_act
from dp2.layers import Sequential
import torch.nn.functional as F
from sg3_torch_utils.ops import fma, upfirdn2d
from resize_right import resize as resize_r
from .utils import spatial_embed_keypoints


class Upfirdn2d(torch.nn.Module):
    def __init__(self, down=1, up=1):
        super().__init__()
        self.register_buffer("resample_filter", upfirdn2d.setup_filter([1, 3, 3, 1]))
        fw, fh = upfirdn2d._get_filter_size(self.resample_filter)
        px0, px1, py0, py1 = upfirdn2d._parse_padding(0)
        self.down = down
        self.up = up
        if up > 1:
            px0 += (fw + up - 1) // 2
            px1 += (fw - up) // 2
            py0 += (fh + up - 1) // 2
            py1 += (fh - up) // 2
        if down > 1:
            px0 += (fw - down + 1) // 2
            px1 += (fw - down) // 2
            py0 += (fh - down + 1) // 2
            py1 += (fh - down) // 2
        self.padding = [px0, px1, py0, py1]
        self.gain = up**2
        assert up <= 2

    def forward(self, x, *args):
        if isinstance(x, dict):
            x = {k: v for k, v in x.items()}
            x["x"] = upfirdn2d.upfirdn2d(
                x["x"],
                self.resample_filter,
                down=self.down,
                padding=self.padding,
                up=self.up,
                gain=self.gain,
            )
            return x
        x = upfirdn2d.upfirdn2d(
            x,
            self.resample_filter,
            down=self.down,
            padding=self.padding,
            up=self.up,
            gain=self.gain,
        )
        if len(args) == 0:
            return x
        return (x, *args)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.repr = dict(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation=activation,
            lr_multiplier=lr_multiplier,
            bias_init=bias_init,
        )
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            if self.bias_gain != 1:
                b = b * self.bias_gain
        x = F.linear(x, w)
        x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation="lrelu",
        bias=True,
        norm=None,
        lr_multiplier=1,
        bias_init=0,
        w_dim=None,
        gain=1,
        padding=None,
        resolution=None,
        use_noise=False,
    ):
        super().__init__()
        if norm == torch.nn.InstanceNorm2d:
            self.norm = torch.nn.InstanceNorm2d(None)
        elif isinstance(norm, torch.nn.Module):
            self.norm = norm
        elif norm == "gnorm":
            self.norm = nn.GroupNorm(32, in_channels)
        elif norm:
            self.norm = torch.nn.InstanceNorm2d(None)
        elif norm == False:
            pass
        elif norm is not None:
            raise ValueError(f"norm not supported: {norm}")
        self.activation = activation
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = kernel_size // 2 if padding is None else padding
        self.repr = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias,
        )
        self.gain = gain
        self.register_buffer(
            "act_gain",
            torch.tensor(bias_act.activation_funcs[activation].def_gain).float(),
            persistent=False,
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_channels * (kernel_size**2))
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros([out_channels]) + bias_init)
            if bias
            else None
        )
        self.bias_gain = lr_multiplier
        if w_dim is not None:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.affine_beta = FullyConnectedLayer(w_dim, in_channels, bias_init=0)

        if use_noise:
            self.resolution = resolution
            self.register_buffer("noise_const", torch.randn(list(resolution)))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.use_noise = use_noise

    def forward(self, x, w=None, s=None):
        # Implements normalization -> mod ->  conv -> activatio
        if hasattr(self, "norm"):
            x = self.norm(x)
        if hasattr(self, "affine") and s is None:
            gamma = self.affine(w).view(-1, self.in_channels, 1, 1)
            beta = self.affine_beta(w).view(-1, self.in_channels, 1, 1)
            x = fma.fma(x, gamma, beta)
        elif hasattr(self, "affine"):
            s = next(s)
            s = s[: self.in_channels * 2]
            gamma, beta = s.view(
                1,
                -1,
                1,
                1,
            ).chunk(2, dim=1)
            x = fma.fma(x, gamma, beta)
        act_gain = self.act_gain
        x = bias_act.bias_act(x, None, act=self.activation, gain=act_gain)
        b = self.bias * self.bias_gain * self.gain if self.bias is not None else None
        w = self.weight * self.weight_gain * self.gain
        x = F.conv2d(input=x, weight=w, padding=self.padding, bias=b)

        if self.training and self.use_noise:
            noise = (
                torch.randn([x.shape[0], 1, *self.resolution], device=x.device)
                * self.noise_strength
            )
        elif self.use_noise:
            noise = self.noise_const * self.noise_strength
        if self.use_noise:
            x = x + noise
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])


class SG2ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels: List[int],  # Number of output channels.
        skip_gain=np.sqrt(0.5),
        layer_scale=False,
        **layer_kwargs,  # Arguments for conv layer.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv0 = Conv2d(in_channels, out_channels[0], **layer_kwargs)
        self.conv1 = Conv2d(
            out_channels[0], out_channels[1], **layer_kwargs, gain=skip_gain
        )
        self.skip = Conv2d(
            in_channels,
            out_channels[1],
            kernel_size=1,
            bias=False,
            gain=skip_gain,
            activation="linear",
        )
        if layer_scale:
            self.layer_scale = nn.Parameter(
                torch.zeros((1, out_channels[1], 1, 1), dtype=torch.float32) + 1e-5
            )

    def forward(self, x, w=None, **layer_kwargs):
        y = x
        if hasattr(self, "skip"):
            y = self.skip(x)
        x = self.conv0(x, w, **layer_kwargs)
        x = self.conv1(x, w, **layer_kwargs)
        if hasattr(self, "layer_scale"):
            x = x * self.layer_scale
        return y + x


def cast_tuple(val, length=None):
    if isinstance(val, Iterable) and not isinstance(val, str):
        val = tuple(val)
    if not isinstance(val, tuple):
        val = (val,) * length
    assert len(val) == length, (val, length)
    return val


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_mults: Tuple[int],
        num_resnet_blocks,  # Number of resnet blocks per resolution
        w_dim: int,
        norm_enc: bool,
        imsize: Tuple[int],
        use_noise: bool,
        layer_scale: bool,
    ) -> None:
        super().__init__()
        self.n_layers = len(dim_mults)
        n_layers = len(dim_mults)
        dims = [dim * m for m in dim_mults]
        enc_blk = partial(
            SG2ResidualBlock,
            norm=norm_enc,
            w_dim=w_dim,
            use_noise=use_noise,
            layer_scale=layer_scale,
        )
        layers = []
        # Currently up/down sampling is done by bilinear upsampling.
        # This can be simplified by replacing it with a strided upsampling layer...

        for lidx in range(n_layers):
            resolution = [r // (2**lidx) for r in imsize]

            dim_in = dims[lidx]
            dim_out = dims[min(lidx + 1, n_layers - 1)]
            res_blocks = nn.ModuleList()
            for i in range(num_resnet_blocks[lidx]):
                cur_dim = dim_out if i == num_resnet_blocks[lidx] - 1 else dim_in
                gain = np.sqrt(1 / 2)
                block = enc_blk(
                    dim_in,
                    [dim_in, cur_dim],
                    skip_gain=gain,
                    resolution=resolution,
                )
                res_blocks.append(block)
            layers.append(res_blocks)
        layers.reverse()
        self.layers = nn.ModuleList(layers)

        self.downsample = Upfirdn2d(down=2)

    def forward(self, x, w, s):
        unet_features = []
        for lidx, res_blocks in enumerate(self.layers[::-1]):
            is_last = lidx == len(self.layers) - 1
            for block in res_blocks:
                x = block(x, w=w, s=s)
                unet_features.append(x)
            if not is_last:
                x = self.downsample(x)
        return x, unet_features


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_mults: tuple,
        num_resnet_blocks,  # Number of resnet blocks per resolution
        n_middle_blocks: int,
        w_dim: int,
        norm_dec: str,
        im_channels: int,
        imsize: Tuple[int],
        use_noise: bool,
        layer_scale: bool,
    ) -> None:
        super().__init__()
        dims = [dim * m for m in dim_mults]
        n_layers = len(dims)
        # initialize decoder
        self.layers = nn.ModuleList()
        dec_blk = partial(
            SG2ResidualBlock,
            norm=norm_dec,
            w_dim=w_dim,
            use_noise=use_noise,
            layer_scale=layer_scale,
        )
        self.unet_layers = torch.nn.ModuleList()
        for lidx in range(n_layers):  # Iterate from lowest resolution
            dim_in = dims[min(-lidx, -1)]
            dim_out = dims[-1 - lidx]
            res_blocks = nn.ModuleList()
            unet_skips = nn.ModuleList()
            for i in range(num_resnet_blocks[-lidx - 1]):
                resolution = [r//(2**(n_layers-1-lidx)) for r in imsize]
                is_first = i == 0
                has_unet = is_first and lidx != 0
                cur_dim = dim_in if is_first else dim_out
                n = 2
                n += int(has_unet)
                gain = np.sqrt(1 / n)
                block = dec_blk(
                    cur_dim, [cur_dim, dim_out], skip_gain=gain, resolution=resolution
                )
                res_blocks.append(block)
                if has_unet:
                    unet_block = Conv2d(
                        cur_dim,
                        cur_dim,
                        kernel_size=1,
                        norm=True,
                        gain=gain,
                        bias=False,
                    )
                    unet_skips.append(unet_block)
                else:
                    unet_skips.append(torch.nn.Identity())
            self.layers.append(res_blocks)
            self.unet_layers.append(unet_skips)
        self.upsample = Upfirdn2d(up=2)

        middle_blocks = []
        for i in range(n_middle_blocks):
            block = dec_blk(
                dims[-1],
                [dims[-1], dims[-1]],
                skip_gain=np.sqrt(0.5),
                resolution=[r // (2 ** (n_layers - 1)) for r in imsize],
            )
            middle_blocks.append(block)
        if n_middle_blocks != 0:
            self.middle_blocks = Sequential(*middle_blocks)
        self.to_rgb = Conv2d(dim_out, im_channels, 1, activation="linear")
        self.im_channels = im_channels

    def forward(self, x, w, unet_features, s):
        if hasattr(self, "middle_blocks"):
            x = self.middle_blocks(x, w=w, s=s)
        features = []
        unet_features = iter(reversed(unet_features))
        for i, (unet_skip, res_blocks) in enumerate(zip(self.unet_layers, self.layers)):
            is_last = i == len(self.layers) - 1
            for (
                skip,
                block,
            ) in zip(unet_skip, res_blocks):
                skip_x = next(unet_features)
                if not isinstance(skip, torch.nn.Identity):
                    skip_x = skip(skip_x)
                    x = x + skip_x
                x = block(x, w=w, s=s)
            features.append(x)
            if not is_last:
                x = self.upsample(x)
        x = self.to_rgb(x)
        return x, features


class TriaGAN(BaseStyleGAN):
    def __init__(
        self,
        imsize: Tuple[int],
        im_channels: int,
        dim: int,
        dim_mults: tuple,
        num_resnet_blocks,  # Number of resnet blocks per resolution
        n_middle_blocks: int,
        z_channels: int,
        w_dim: int,
        norm_enc: bool,
        norm_dec: str,
        use_maskrcnn_mask: bool,
        input_keypoints: bool,
        n_keypoints: int,
        input_keypoint_indices: Tuple[int],
        style_net: nn.Module,
        input_joint=False,
        latent_space=None,
        use_noise=True,
        layer_scale=False,
        **kwargs,
    ) -> None:
        super().__init__(z_channels, w_dim, **kwargs)
        self.style_net = style_net
        self.n_keypoints = n_keypoints
        self.input_keypoint_indices = list(input_keypoint_indices)
        self.input_keypoints = input_keypoints
        self.imsize = tuple(imsize)
        self.input_joint = input_joint
        self.dim = dim
        self.dim_mults = dim_mults
        if latent_space is not None:
            self.latent_space = latent_space
        assert self.latent_space in ["W", "W_kp", "Z"]
        n_layers = len(dim_mults)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, n_layers)
        self.from_rgb = Conv2d(
            im_channels
            + 2
            + 2 * int(use_maskrcnn_mask)
            + self.input_keypoints * len(input_keypoint_indices)
            + input_joint * 6,
            dim,
            7,
            activation="linear",
        )
        self.from_rgb.weight_gain = 1 / np.sqrt(
            (3 + 1 + use_maskrcnn_mask + self.input_keypoints + input_joint) * (7**2)
        )

        self.use_maskrcnn_mask = use_maskrcnn_mask
        self.encoder = Encoder(
            dim,
            dim_mults,
            num_resnet_blocks,
            w_dim,
            norm_enc,
            imsize=imsize,
            layer_scale=layer_scale,
            use_noise=False
        )
        self.decoder = Decoder(
            dim,
            dim_mults,
            num_resnet_blocks,
            n_middle_blocks,
            w_dim,
            norm_dec,
            im_channels=im_channels,
            use_noise=use_noise,
            imsize=imsize,
            layer_scale=layer_scale,
        )

    def get_input(
        self, condition, mask, maskrcnn_mask, keypoints, joint_map, **kwargs
    ) -> torch.Tensor:
        if self.use_maskrcnn_mask:
            x = torch.cat(
                (condition, mask, 1 - mask, maskrcnn_mask, 1 - maskrcnn_mask), dim=1
            )
        else:
            x = torch.cat((condition, mask, 1 - mask), dim=1)

        if self.input_keypoints:
            keypoints = keypoints[:, self.input_keypoint_indices]
            one_hot_pose = spatial_embed_keypoints(keypoints, x).to(x.device)
            x = torch.cat((x, one_hot_pose), dim=1)
        if self.input_joint:
            joint_map_oh = torch.zeros(
                (joint_map.shape[0], 6 + 1, *joint_map.shape[-2:]),
                device=joint_map.device,
                dtype=torch.float32,
            )
            joint_map_oh.scatter_(1, joint_map.long(), 1)
            x = torch.cat((x, joint_map_oh[:, 1:]), dim=1)  # Zero is BG
        return x

    def forward(
        self,
        condition,
        mask,
        maskrcnn_mask=None,
        img=None,
        z=None,
        w=None,
        update_emas=False,
        keypoints=None,
        s=None,
        joint_map=None,
        **kwargs,
    ):
        if z is None:
            z = self.get_z(condition)

        if w is None and s is None:
            w = self.style_net(z, keypoints=keypoints, update_emas=update_emas)
        x = self.get_input(
            condition, mask, maskrcnn_mask, keypoints, joint_map, **kwargs
        )
        x = self.from_rgb(x)
        x, unet_features = self.encoder(x, w, s=s)

        x, _ = self.decoder(x, w, unet_features, s=s)
        unmasked = x
        x = mask * img + (1 - mask) * x
        out = dict(img=x, unmasked=unmasked)
        return out

    def get_w(self, z, update_emas, keypoints=None):
        return self.style_net(z, update_emas=update_emas, keypoints=keypoints)
