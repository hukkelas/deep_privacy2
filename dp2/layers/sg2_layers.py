from typing import List
import numpy as np
import torch
import tops
import torch.nn.functional as F
from sg3_torch_utils.ops import conv2d_resample
from sg3_torch_utils.ops import upfirdn2d
from sg3_torch_utils.ops import bias_act
from sg3_torch_utils.ops.fma import fma


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.repr = dict(
            in_features=in_features, out_features=out_features, bias=bias,
            activation=activation, lr_multiplier=lr_multiplier, bias_init=bias_init)
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        x = F.linear(x, w)
        x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size=3,  # Convolution kernel size.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 bias=True,
                 norm=False,
                 lr_multiplier=1,
                 bias_init=0,
                 w_dim=None,
                 gain=1,
                 ):
        super().__init__()
        if norm:
            self.norm = torch.nn.InstanceNorm2d(None)
        assert norm in [True, False]
        self.up = up
        self.down = down
        self.activation = activation
        self.conv_clamp = conv_clamp if conv_clamp is None else conv_clamp * gain
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = kernel_size // 2

        self.repr = dict(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, up=up, down=down,
            activation=activation, resample_filter=resample_filter, conv_clamp=conv_clamp, bias=bias,
        )

        if self.up == 1 and self.down == 1:
            self.resample_filter = None
        else:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.act_gain = bias_act.activation_funcs[activation].def_gain * gain
        self.weight_gain = lr_multiplier / np.sqrt(in_channels * (kernel_size ** 2))
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]) + bias_init) if bias else None
        self.bias_gain = lr_multiplier
        if w_dim is not None:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.affine_beta = FullyConnectedLayer(w_dim, in_channels, bias_init=0)

    def forward(self, x, w=None, s=None):
        tops.assert_shape(x, [None, self.weight.shape[1], None, None])
        if s is not None:
            s = s[..., :self.in_channels * 2]
            gamma, beta = s.view(-1, self.in_channels * 2, 1, 1).chunk(2, dim=1)
            x = fma(x, gamma, beta)
        elif hasattr(self, "affine"):
            gamma = self.affine(w).view(-1, self.in_channels, 1, 1)
            beta = self.affine_beta(w).view(-1, self.in_channels, 1, 1)
            x = fma(x, gamma, beta)
        w = self.weight * self.weight_gain
        # Removing flip weight is not safe.
        x = conv2d_resample.conv2d_resample(x, w, self.resample_filter, self.up,
                                            self.down, self.padding, flip_weight=self.up == 1)
        if hasattr(self, "norm"):
            x = self.norm(x)
        b = self.bias * self.bias_gain if self.bias is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=self.act_gain, clamp=self.conv_clamp)
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])


class Block(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 up=1,
                 down=1,
                 **layer_kwargs,  # Arguments for SynthesisLayer.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.down = down
        self.conv0 = Conv2d(in_channels, out_channels, down=down, conv_clamp=conv_clamp, **layer_kwargs)
        self.conv1 = Conv2d(out_channels, out_channels, up=up, conv_clamp=conv_clamp, **layer_kwargs)

    def forward(self, x, **layer_kwargs):
        x = self.conv0(x, **layer_kwargs)
        x = self.conv1(x, **layer_kwargs)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 up=1,
                 down=1,
                 gain_out=np.sqrt(0.5),
                 fix_residual: bool = False,
                 **layer_kwargs,  # Arguments for conv layer.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down = down
        self.conv0 = Conv2d(in_channels, out_channels, down=down, conv_clamp=conv_clamp, **layer_kwargs)

        self.conv1 = Conv2d(out_channels, out_channels, up=up, conv_clamp=conv_clamp, gain=gain_out, **layer_kwargs)

        self.skip = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False, up=up, down=down,
            activation="linear" if fix_residual else "lrelu",
            gain=gain_out
        )
        self.gain_out = gain_out

    def forward(self, x, w=None, s=None, **layer_kwargs):
        y = self.skip(x)
        s_ = next(s) if s is not None else None
        x = self.conv0(x, w, s=s_, **layer_kwargs)
        s_ = next(s) if s is not None else None
        x = self.conv1(x, w, s=s_, **layer_kwargs)
        x = y + x
        return x


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with tops.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 resolution: List[int],  # Resolution of this block.
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2d(
            in_channels + mbstd_num_channels, in_channels,
            kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * resolution[0] * resolution[1], in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, x):
        tops.assert_shape(x, [None, self.in_channels, *self.resolution])  # [NCHW]
        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        return x
