import torch
import torch.nn as nn
from easydict import EasyDict
from .base import BaseGenerator
import numpy as np
from typing import List


class LatentVariableConcat(nn.Module):

    def __init__(self, conv2d_config):
        super().__init__()

    def forward(self, _inp):
        x, mask, batch = _inp
        z = batch["z"]
        x = torch.cat((x, z), dim=1)
        return (x, mask, batch)


def get_padding(kernel_size: int, dilation: int, stride: int):
    out = (dilation * (kernel_size - 1) - 1) / 2 + 1
    return int(np.floor(out))


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 demodulation=False, wsconv=False, gain=1,
                 *args, **kwargs):
        if padding is None:
            padding = get_padding(kernel_size, dilation, stride)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.demodulation = demodulation
        self.wsconv = wsconv
        if self.wsconv:
            fan_in = np.prod(self.weight.shape[1:]) / self.groups
            self.ws_scale = gain / np.sqrt(fan_in)
            nn.init.normal_(self.weight)
        if bias:
            nn.init.constant_(self.bias, val=0)
        assert not self.padding_mode == "circular",\
            "conv2d_forward does not support circular padding. Look at original pytorch code"

    def _get_weight(self):
        weight = self.weight
        if self.wsconv:
            weight = self.ws_scale * weight
        if self.demodulation:
            demod = torch.rsqrt(weight.pow(2).sum([1, 2, 3]) + 1e-7)
            weight = weight * demod.view(self.out_channels, 1, 1, 1)
        return weight

    def conv2d_forward(self, x, weight, bias=True):
        bias_ = None
        if bias:
            bias_ = self.bias
        return nn.functional.conv2d(x, weight, bias_, self.stride,
                                    self.padding, self.dilation, self.groups)

    def forward(self, _inp):
        x, mask = _inp
        weight = self._get_weight()
        return self.conv2d_forward(x, weight), mask

    def __repr__(self):
        return ", ".join([
            super().__repr__(),
            f"Demodulation={self.demodulation}",
            f"Weight Scale={self.wsconv}",
            f"Bias={self.bias is not None}"
        ])


class LeakyReLU(nn.LeakyReLU):

    def forward(self, _inp):
        x, mask = _inp
        return super().forward(x), mask


class AvgPool2d(nn.AvgPool2d):

    def forward(self, _inp):
        x, mask, *args = _inp
        x = super().forward(x)
        mask = super().forward(mask)
        if len(args) > 0:
            return (x, mask, *args)
        return x, mask


def up(x):
    if x.shape[0] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
        # Analytical normalization
        return x
    return nn.functional.interpolate(
        x, scale_factor=2, mode="nearest")


class NearestUpsample(nn.Module):

    def forward(self, _inp):
        x, mask, *args = _inp
        x = up(x)
        mask = up(mask)
        if len(args) > 0:
            return (x, mask, *args)
        return x, mask


class PixelwiseNormalization(nn.Module):

    def forward(self, _inp):
        x, mask = _inp
        norm = torch.rsqrt((x**2).mean(dim=1, keepdim=True) + 1e-7)
        return x * norm, mask


class Linear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        fanIn = in_features
        self.wtScale = 1 / np.sqrt(fanIn)

        nn.init.normal_(self.weight)
        nn.init.constant_(self.bias, val=0)

    def _get_weight(self):
        return self.weight * self.wtScale

    def forward_linear(self, x, weight):
        return nn.functional.linear(x, weight, self.bias)

    def forward(self, x):
        return self.forward_linear(x, self._get_weight())


class OneHotPoseConcat(nn.Module):

    def forward(self, _inp):
        x, mask, batch = _inp
        landmarks = batch["landmarks_oh"]
        res = x.shape[-1]
        landmark = landmarks[res]
        x = torch.cat((x, landmark), dim=1)
        del batch["landmarks_oh"][res]
        return x, mask, batch


def transition_features(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape,\
        "Old shape: {}, New: {}".format(x_old.shape, x_new.shape)
    return torch.lerp(x_old.float(), x_new.float(), transition_variable)


class TransitionBlock(nn.Module):

    def forward(self, _inp):
        x, mask, batch = _inp
        x = transition_features(
            batch["x_old"], x, batch["transition_value"])
        mask = transition_features(
            batch["mask_old"], mask, batch["transition_value"])
        del batch["x_old"]
        del batch["mask_old"]
        return x, mask, batch


class UnetSkipConnection(nn.Module):

    def __init__(self, conv2d_config: dict, in_channels: int,
                 out_channels: int, resolution: int,
                 residual: bool, enabled: bool):
        super().__init__()
        self.use_iconv = conv2d_config.conv.type == "iconv"
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._resolution = resolution
        self._enabled = enabled
        self._residual = residual
        if self.use_iconv:
            self.beta0 = torch.nn.Parameter(torch.tensor(1.))
            self.beta1 = torch.nn.Parameter(torch.tensor(1.))
        else:
            if self._residual:
                self.conv = build_base_conv(
                    conv2d_config, False, in_channels // 2,
                    out_channels, kernel_size=1, padding=0)
            else:
                self.conv = ConvAct(
                    conv2d_config, in_channels, out_channels,
                    kernel_size=1, padding=0)

    def forward(self, _inp):
        if not self._enabled:
            return _inp
        x, mask, batch = _inp
        skip_x, skip_mask = batch["unet_features"][self._resolution]
        assert x.shape == skip_x.shape, (x.shape, skip_x.shape)
        del batch["unet_features"][self._resolution]
        if self.use_iconv:
            denom = skip_mask * self.beta0.relu() + mask * self.beta1.relu() + 1e-8
            gamma = skip_mask * self.beta0.relu() / denom
            x = skip_x * gamma + (1 - gamma) * x
            mask = skip_mask * gamma + (1 - gamma) * mask
        else:
            if self._residual:
                skip_x, skip_mask = self.conv((skip_x, skip_mask))
                x = (x + skip_x) / np.sqrt(2)
                if self._probabilistic:
                    mask = (mask + skip_mask) / np.sqrt(2)
            else:
                x = torch.cat((x, skip_x), dim=1)
                x, mask = self.conv((x, mask))
        return x, mask, batch

    def __repr__(self):
        return " ".join([
            self.__class__.__name__,
            f"In channels={self._in_channels}",
            f"Out channels={self._out_channels}",
            f"Residual: {self._residual}",
            f"Enabled: {self._enabled}"
            f"IConv: {self.use_iconv}"
        ])


def get_conv(ctype, post_act):
    type2conv = {
        "conv": Conv2d,
        "gconv": GatedConv
    }
    # Do not apply for output layer
    if not post_act and ctype in ["gconv", "iconv"]:
        return type2conv["conv"]
    assert ctype in type2conv
    return type2conv[ctype]


def build_base_conv(
        conv2d_config, post_act: bool, *args, **kwargs) -> nn.Conv2d:
    for k, v in conv2d_config.conv.items():
        assert k not in kwargs
        kwargs[k] = v
    # Demodulation should not be used for output layers.
    demodulation = conv2d_config.normalization == "demodulation" and post_act
    kwargs["demodulation"] = demodulation
    conv = get_conv(conv2d_config.conv.type, post_act)
    return conv(*args, **kwargs)


def build_post_activation(in_channels, conv2d_config) -> List[nn.Module]:
    _layers = []
    negative_slope = conv2d_config.leaky_relu_nslope
    _layers.append(LeakyReLU(negative_slope, inplace=True))
    if conv2d_config.normalization == "pixel_wise":
        _layers.append(PixelwiseNormalization())
    return _layers


def build_avgpool(conv2d_config, kernel_size) -> nn.AvgPool2d:
    return AvgPool2d(kernel_size)


def build_convact(conv2d_config, *args, **kwargs):
    conv = build_base_conv(conv2d_config, True, *args, **kwargs)
    out_channels = conv.out_channels
    post_act = build_post_activation(out_channels, conv2d_config)
    return nn.Sequential(conv, *post_act)


class ConvAct(nn.Module):

    def __init__(self, conv2d_config, *args, **kwargs):
        super().__init__()
        self._conv2d_config = conv2d_config
        conv = build_base_conv(conv2d_config, True, *args, **kwargs)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        _layers = [conv]
        _layers.extend(build_post_activation(self.out_channels, conv2d_config))
        self.layers = nn.Sequential(*_layers)

    def forward(self, _inp):
        return self.layers(_inp)


class GatedConv(Conv2d):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        out_channels *= 2
        super().__init__(in_channels, out_channels, *args, **kwargs)
        assert self.out_channels % 2 == 0
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def conv2d_forward(self, x, weight, bias=True):
        x_ = super().conv2d_forward(x, weight, bias)
        x = x_[:, :self.out_channels // 2]
        y = x_[:, self.out_channels // 2:]
        x = self.lrelu(x)
        y = y.sigmoid()
        assert x.shape == y.shape, f"{x.shape}, {y.shape}"
        return x * y


class BasicBlock(nn.Module):

    def __init__(
            self, conv2d_config, resolution: int, in_channels: int,
            out_channels: List[int], residual: bool):
        super().__init__()
        assert len(out_channels) == 2
        self._resolution = resolution
        self._residual = residual
        self.out_channels = out_channels
        _layers = []
        _in_channels = in_channels
        for out_ch in out_channels:
            conv = build_base_conv(
                conv2d_config, True, _in_channels, out_ch, kernel_size=3,
                resolution=resolution)
            _layers.append(conv)
            _layers.extend(build_post_activation(_in_channels, conv2d_config))
            _in_channels = out_ch
        self.layers = nn.Sequential(*_layers)
        if self._residual:
            self.residual_conv = build_base_conv(
                conv2d_config, post_act=False, in_channels=in_channels,
                out_channels=out_channels[-1],
                kernel_size=1, padding=0)
            self.const = 1 / np.sqrt(2)

    def forward(self, _inp):
        x, mask, batch = _inp
        y = x
        mask_ = mask
        assert y.shape[-1] == self._resolution or y.shape[-1] == 1
        y, mask = self.layers((x, mask))
        if self._residual:
            residual, mask_ = self.residual_conv((x, mask_))
            y = (y + residual) * self.const
            mask = (mask + mask_) * self.const
        return y, mask, batch

    def extra_repr(self):
        return f"Residual={self._residual}, Resolution={self._resolution}"


class PoseNormalize(nn.Module):

    @torch.no_grad()
    def forward(self, x):
        return x * 2 - 1


class ScalarPoseFCNN(nn.Module):

    def __init__(self, pose_size, hidden_size,
                 output_shape):
        super().__init__()
        pose_size = pose_size
        self._hidden_size = hidden_size
        output_size = np.prod(output_shape)
        self.output_shape = output_shape
        self.pose_preprocessor = nn.Sequential(
            PoseNormalize(),
            Linear(pose_size, hidden_size),
            nn.LeakyReLU(.2),
            Linear(hidden_size, output_size),
            nn.LeakyReLU(.2)
        )

    def forward(self, _inp):
        x, mask, batch = _inp
        pose_info = batch["landmarks"]
        del batch["landmarks"]
        pose = self.pose_preprocessor(pose_info)
        pose = pose.view(-1, *self.output_shape)
        if x.shape[0] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
            # Analytical normalization propagation
            pose = pose.mean(dim=2, keepdim=True).mean(dim=3, keepdims=True)
        x = torch.cat((x, pose), dim=1)
        return x, mask, batch

    def __repr__(self):
        return " ".join([
            self.__class__.__name__,
            f"hidden_size={self._hidden_size}",
            f"output shape={self.output_shape}"
        ])


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # Channel multiplier
        self.in_channels = in_channels
        self.theta = Conv2d(
            self.in_channels, self.in_channels // 8, kernel_size=1, padding=0,
            bias=False)
        self.phi = Conv2d(
            self.in_channels, self.in_channels // 8, kernel_size=1, padding=0,
            bias=False)
        self.g = Conv2d(
            self.in_channels, self.in_channels // 2, kernel_size=1, padding=0,
            bias=False)
        self.o = Conv2d(
            self.in_channels // 2, self.in_channels, kernel_size=1, padding=0,
            bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, _inp):
        x, mask, batch = _inp
        # Apply convs
        theta, _ = self.theta((x, None))
        phi = nn.functional.max_pool2d(self.phi((x, None))[0], [2, 2])
        g = nn.functional.max_pool2d(self.g((x, None))[0], [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = nn.functional.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path

        o = self.o((torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                            self.in_channels // 2, x.shape[2], x.shape[3]), None))[0]
        return self.gamma * o + x, mask, batch


class MSGGenerator(BaseGenerator):

    def __init__(self):
        super().__init__(512)
        max_imsize = 128
        unet = dict(enabled=True, residual=False)

        min_fmap_resolution = 4
        model_size = 512
        image_channels = 3
        pose_size = 14
        residual = False
        conv_size = {
            4: model_size,
            8: model_size,
            16: model_size,
            32: model_size,
            64: model_size//2,
            128: model_size//4,
            256: model_size//8,
            512: model_size//16
        }
        self.removable_hooks = []
        self.rgb_convolutions = nn.ModuleDict()
        self.max_imsize = max_imsize
        self._image_channels = image_channels
        self._min_fmap_resolution = min_fmap_resolution
        self._residual = residual
        self._pose_size = pose_size
        self.current_imsize = max_imsize
        self._unet_cfg = unet
        self.concat_input_mask = True
        self.res2channels = {int(k): v for k, v in conv_size.items()}

        self.conv2d_config = EasyDict(
            pixel_normalization=True,
            leaky_relu_nslope=.2,
            normalization="pixel_wise",
            conv=dict(
                type="conv",
                wsconv=True,
                gain=1,
            )
        )
        self._init_decoder()
        self._init_encoder()

    def _init_encoder(self):
        self.encoder = nn.ModuleList()
        imsize = self.max_imsize
        self.from_rgb = build_convact(
            self.conv2d_config,
            in_channels=self._image_channels + self.concat_input_mask*2,
            out_channels=self.res2channels[imsize],
            kernel_size=1)
        while imsize >= self._min_fmap_resolution:
            current_size = self.res2channels[imsize]
            next_size = self.res2channels[max(imsize//2, self._min_fmap_resolution)]
            block = BasicBlock(
                self.conv2d_config, imsize, current_size,
                [current_size, next_size], self._residual)
            self.encoder.add_module(f"basic_block{imsize}", block)
            if imsize != self._min_fmap_resolution:
                self.encoder.add_module(
                    f"downsample{imsize}", AvgPool2d(2))
            imsize //= 2

    def _init_decoder(self):
        self.decoder = nn.ModuleList()
        self.decoder.add_module(
            "latent_concat", LatentVariableConcat(self.conv2d_config))
        if self._pose_size > 0:
            m = self._min_fmap_resolution
            pose_shape = (16, m, m)
            pose_fcnn = ScalarPoseFCNN(self._pose_size, 128, pose_shape)
            self.decoder.add_module("pose_fcnn", pose_fcnn)
        imsize = self._min_fmap_resolution
        self.rgb_convolutions = nn.ModuleDict()
        while imsize <= self.max_imsize:
            current_size = self.res2channels[max(imsize//2, self._min_fmap_resolution)]
            start_size = current_size
            if imsize == self._min_fmap_resolution:
                start_size += 32
                if self._pose_size > 0:
                    start_size += 16
            else:
                self.decoder.add_module(f"upsample{imsize}", NearestUpsample())
                skip = UnetSkipConnection(
                    self.conv2d_config, current_size*2, current_size, imsize,
                    **self._unet_cfg)
                self.decoder.add_module(f"skip_connection{imsize}", skip)
            next_size = self.res2channels[imsize]
            block = BasicBlock(
                self.conv2d_config, imsize, start_size, [start_size, next_size],
                residual=self._residual)
            self.decoder.add_module(f"basic_block{imsize}", block)

            to_rgb = build_base_conv(
                self.conv2d_config, False, in_channels=next_size,
                out_channels=self._image_channels, kernel_size=1)
            self.rgb_convolutions[str(imsize)] = to_rgb
            imsize *= 2
        self.norm_constant = len(self.rgb_convolutions)

    def forward_decoder(self, x, mask, batch):
        imsize_start = max(x.shape[-1] // 2, 1)
        rgb = torch.zeros(
            (x.shape[0], self._image_channels,
             imsize_start, imsize_start),
            dtype=x.dtype, device=x.device)
        mask_size = 1
        mask_out = torch.zeros(
            (x.shape[0], mask_size,
             imsize_start, imsize_start),
            dtype=x.dtype, device=x.device)
        imsize = self._min_fmap_resolution // 2
        for module in self.decoder:
            x, mask, batch = module((x, mask, batch))
            if isinstance(module, BasicBlock):
                imsize *= 2
                rgb = up(rgb)
                mask_out = up(mask_out)
                conv = self.rgb_convolutions[str(imsize)]
                rgb_, mask_ = conv((x, mask))
                assert rgb_.shape == rgb.shape,\
                    f"rgb_ {rgb_.shape}, rgb: {rgb.shape}"
                rgb = rgb + rgb_
        return rgb / self.norm_constant, mask_out

    def forward_encoder(self, x, mask, batch):
        if self.concat_input_mask:
            x = torch.cat((x, mask, 1 - mask), dim=1)
        unet_features = {}
        x, mask = self.from_rgb((x, mask))
        for module in self.encoder:
            x, mask, batch = module((x, mask, batch))
            if isinstance(module, BasicBlock):
                unet_features[module._resolution] = (x, mask)
        return x, mask, unet_features

    def forward(
            self,
            condition,
            mask, keypoints=None, z=None,
            **kwargs):
        keypoints = keypoints[:, :, :2].flatten(start_dim=1).clip(-1, 1)
        if z is None:
            z = self.get_z(condition)
        z = z.view(-1, 32, 4, 4)
        batch = dict(
            landmarks=keypoints,
            z=z)
        orig_mask = mask
        x, mask, unet_features = self.forward_encoder(condition, mask, batch)
        batch = dict(
            landmarks=keypoints,
            z=z,
            unet_features=unet_features)
        x, mask = self.forward_decoder(x, mask, batch)
        x = condition * orig_mask + (1 - orig_mask) * x
        return dict(img=x)

    def load_state_dict(self, state_dict, strict=True):
        if "parameters" in state_dict:
            state_dict = state_dict["parameters"]
        old_checkpoint = any("basic_block0" in key for key in state_dict)
        if not old_checkpoint:
            return super().load_state_dict(state_dict, strict=strict)
        mapping = {}
        imsize = self._min_fmap_resolution
        i = 0
        while imsize <= self.max_imsize:
            old_key = f"decoder.basic_block{i}."
            new_key = f"decoder.basic_block{imsize}."
            mapping[old_key] = new_key
            if i >= 1:
                old_key = old_key.replace("basic_block", "skip_connection")
                new_key = new_key.replace("basic_block", "skip_connection")
                mapping[old_key] = new_key
            mapping[old_key] = new_key
            old_key = f"encoder.basic_block{i}."
            new_key = f"encoder.basic_block{imsize}."
            mapping[old_key] = new_key
            old_key = "from_rgb.conv.layers.0."
            new_key = "from_rgb.0."
            mapping[old_key] = new_key
            i += 1
            imsize *= 2
        new_sd = {}
        for key, value in state_dict.items():
            old_key = key
            if "from_rgb" in key:
                new_sd[key.replace("encoder.", "").replace(".conv.layers", "")] = value
                continue
            for subkey, new_subkey in mapping.items():
                if subkey in key:
                    old_key = key
                    key = key.replace(subkey, new_subkey)

                    break
            if "decoder.to_rgb" in key:
                continue

            new_sd[key] = value
        return super().load_state_dict(new_sd, strict=strict)

    def update_w(self, *args, **kwargs):
        return
