from sg3_torch_utils.ops import upfirdn2d
import torch
import numpy as np
import torch.nn as nn
from .. import layers
from ..layers.sg2_layers import DiscriminatorEpilogue, ResidualBlock, Block


class SG2Discriminator(layers.Module):

    def __init__(
            self,
            cnum: int,
            max_cnum_mul: int,
            imsize,
            min_fmap_resolution: int,
            im_channels: int,
            input_condition: bool,
            conv_clamp: int,
            input_cse: bool,
            cse_nc: int,
            fix_residual: bool,
    ):
        super().__init__()

        cse_nc = 0 if cse_nc is None else cse_nc
        self._max_imsize = max(imsize)
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self._input_condition = input_condition
        self.input_cse = input_cse
        self.layers = nn.ModuleList()

        out_ch = self.get_chsize(self._max_imsize)
        self.from_rgb = Block(
            im_channels + input_condition*(im_channels+1) + input_cse*(cse_nc+1),
            out_ch, conv_clamp=conv_clamp
        )
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1

        for i in range(n_levels):
            resolution = [x//2**i for x in imsize]
            in_ch = self.get_chsize(max(resolution))
            out_ch = self.get_chsize(max(max(resolution)//2, min_fmap_resolution))

            down = 2
            if i == 0:
                down = 1
            block = ResidualBlock(
                in_ch, out_ch, down=down, conv_clamp=conv_clamp,
                fix_residual=fix_residual
            )
            self.layers.append(block)
        self.output_layer = DiscriminatorEpilogue(
            out_ch, resolution, conv_clamp=conv_clamp)

        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, img, condition, mask, embedding=None, E_mask=None, **kwargs):
        to_cat = [img]
        if self._input_condition:
            to_cat.extend([condition, mask, ])
        if self.input_cse:
            to_cat.extend([embedding, E_mask])
        x = torch.cat(to_cat, dim=1)
        x = self.from_rgb(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.output_layer(x)
        return dict(score=x)

    def get_chsize(self, imsize):
        n = int(np.log2(self._max_imsize) - np.log2(imsize))
        mul = min(2 ** n, self._max_cnum_mul)
        ch = self._cnum * mul
        return int(ch)
