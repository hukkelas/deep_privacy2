import torch
import numpy as np
from dp2.layers import Sequential
from dp2.layers.sg2_layers import Conv2d, FullyConnectedLayer, ResidualBlock
from .base import BaseStyleGAN
from typing import List, Tuple
from .utils import spatial_embed_keypoints, mask_output


def get_chsize(imsize, cnum, max_imsize, max_cnum_mul):
    n = int(np.log2(max_imsize) - np.log2(imsize))
    mul = min(2**n, max_cnum_mul)
    ch = cnum * mul
    return int(ch)


class StyleGANUnet(BaseStyleGAN):
    def __init__(
            self,
            scale_grad: bool,
            im_channels: int,
            min_fmap_resolution: int,
            imsize: List[int],
            cnum: int,
            max_cnum_mul: int,
            mask_output: bool,
            conv_clamp: int,
            input_cse: bool,
            cse_nc: int,
            n_middle_blocks: int,
            input_keypoints: bool,
            n_keypoints: int,
            input_keypoint_indices: Tuple[int],
            fix_errors: bool,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.n_keypoints = n_keypoints
        self.input_keypoint_indices = list(input_keypoint_indices)
        self.input_keypoints = input_keypoints
        assert not (input_cse and input_keypoints)
        cse_nc = 0 if cse_nc is None else cse_nc
        self.imsize = imsize
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self._image_channels = im_channels
        self._max_imsize = max(imsize)
        self.input_cse = input_cse
        self.gain_unet = np.sqrt(1/3)
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        encoder_layers = []
        self.from_rgb = Conv2d(
            im_channels + 1 + input_cse*(cse_nc+1) + input_keypoints*len(self.input_keypoint_indices),
            cnum, 1
        )
        for i in range(n_levels):  # Encoder layers
            resolution = [x//2**i for x in imsize]
            in_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            second_ch = in_ch
            out_ch = get_chsize(max(resolution)//2, cnum, self._max_imsize, max_cnum_mul)
            down = 2

            if i == 0:  # first (lowest) block. Downsampling is performed at the start of the block
                down = 1
            if i == n_levels - 1:
                out_ch = second_ch
            block = ResidualBlock(in_ch, out_ch, down=down, conv_clamp=conv_clamp, fix_residual=fix_errors)
            encoder_layers.append(block)
        self._encoder_out_shape = [
            get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul),
            *resolution]

        self.encoder = torch.nn.ModuleList(encoder_layers)

        # initialize decoder
        decoder_layers = []
        for i in range(n_levels):
            resolution = [x//2**(n_levels-1-i) for x in imsize]
            in_ch = get_chsize(max(resolution)//2, cnum, self._max_imsize, max_cnum_mul)
            out_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            if i == 0:  # first (lowest) block
                in_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)

            up = 1
            if i != n_levels - 1:
                up = 2
            block = ResidualBlock(
                in_ch, out_ch, conv_clamp=conv_clamp, gain_out=np.sqrt(1/3),
                w_dim=self.style_net.w_dim, norm=True, up=up,
                fix_residual=fix_errors
            )
            decoder_layers.append(block)
            if i != 0:
                unet_block = Conv2d(
                    in_ch, in_ch, kernel_size=1, conv_clamp=conv_clamp, norm=True,
                    gain=np.sqrt(1/3) if fix_errors else np.sqrt(.5))
                setattr(self, f"unet_block{i}", unet_block)

        # Initialize "middle blocks" that do not have down/up sample
        middle_blocks = []
        for i in range(n_middle_blocks):
            ch = get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul)
            block = ResidualBlock(
                ch, ch, conv_clamp=conv_clamp, gain_out=np.sqrt(.5) if fix_errors else np.sqrt(1/3),
                w_dim=self.style_net.w_dim, norm=True,
            )
            middle_blocks.append(block)
        if n_middle_blocks != 0:
            self.middle_blocks = Sequential(*middle_blocks)
        self.decoder = torch.nn.ModuleList(decoder_layers)
        self.to_rgb = Conv2d(cnum, im_channels, 1, activation="linear", conv_clamp=conv_clamp)
        # Initialize "middle blocks" that do not have down/up sample
        self.decoder = torch.nn.ModuleList(decoder_layers)
        self.scale_grad = scale_grad
        self.mask_output = mask_output

    def forward_dec(self, x, w, unet_features, condition, mask, s, **kwargs):
        for i, layer in enumerate(self.decoder):
            if i != 0:
                unet_layer = getattr(self, f"unet_block{i}")
                x = x + unet_layer(unet_features[-i])
            x = layer(x, w=w, s=s)
        x = self.to_rgb(x)
        if self.mask_output:
            x = mask_output(True, condition, x, mask)
        return dict(img=x)

    def forward_enc(self, condition, mask, embedding,  keypoints, E_mask, **kwargs):
        if self.input_cse:
            x = torch.cat((condition, mask, embedding, E_mask), dim=1)
        else:
            x = torch.cat((condition, mask), dim=1)
        if self.input_keypoints:
            keypoints = keypoints[:, self.input_keypoint_indices]
            one_hot_pose = spatial_embed_keypoints(keypoints, x)
            x = torch.cat((x, one_hot_pose), dim=1)
        x = self.from_rgb(x)

        unet_features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != len(self.encoder)-1:
                unet_features.append(x)
        if hasattr(self, "middle_blocks"):
            for layer in self.middle_blocks:
                x = layer(x)
        return x, unet_features

    def forward(
            self, condition, mask,
            z=None, embedding=None, w=None, update_emas=False, x=None,
            s=None,
            keypoints=None,
            unet_features=None,
            E_mask=None,
            **kwargs):
        # Used to skip sampling from encoder in inference. E.g. for w projection.
        if x is not None and unet_features is not None:
            assert not self.training
        else:
            x, unet_features = self.forward_enc(condition, mask, embedding, keypoints, E_mask, **kwargs)
        if w is None:
            if z is None:
                z = self.get_z(condition)
            w = self.get_w(z, update_emas=update_emas)
        return self.forward_dec(x, w, unet_features, condition, mask, s, **kwargs)


class ComodStyleUNet(StyleGANUnet):

    def __init__(self, min_comod_res=4, lr_multiplier_comod=1, **kwargs) -> None:
        super().__init__(**kwargs)
        min_fmap = min(self._encoder_out_shape[1:])
        enc_out_ch = self._encoder_out_shape[0]
        n_down = int(np.ceil(np.log2(min_fmap) - np.log2(min_comod_res)))
        comod_layers = []
        in_ch = enc_out_ch
        for i in range(n_down):
            comod_layers.append(Conv2d(enc_out_ch, 256, kernel_size=3, down=2, lr_multiplier=lr_multiplier_comod))
            in_ch = 256
        if n_down == 0:
            comod_layers = [Conv2d(in_ch, 256, kernel_size=3)]
        comod_layers.append(torch.nn.Flatten())
        out_res = [x//2**n_down for x in self._encoder_out_shape[1:]]
        in_ch_fc = np.prod(out_res) * 256
        comod_layers.append(FullyConnectedLayer(in_ch_fc, 512, lr_multiplier=lr_multiplier_comod))
        self.comod_block = Sequential(*comod_layers)
        self.comod_fc = FullyConnectedLayer(
            512+self.style_net.w_dim, self.style_net.w_dim, lr_multiplier=lr_multiplier_comod)

    def forward_dec(self, x, w, unet_features, condition, mask, **kwargs):
        y = self.comod_block(x)
        y = torch.cat((y, w), dim=1)
        y = self.comod_fc(y)
        for i, layer in enumerate(self.decoder):
            if i != 0:
                unet_layer = getattr(self, f"unet_block{i}")
                x = x + unet_layer(unet_features[-i], gain=np.sqrt(.5))
            x = layer(x, w=y)
        x = self.to_rgb(x)
        if self.mask_output:
            x = mask_output(True, condition, x, mask)
        return dict(img=x)

    def get_comod_y(self, batch, w):
        x, unet_features = self.forward_enc(**batch)
        y = self.comod_block(x)
        y = torch.cat((y, w), dim=1)
        y = self.comod_fc(y)
        return y
