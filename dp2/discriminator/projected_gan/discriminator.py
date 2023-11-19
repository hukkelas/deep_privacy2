from typing import List
import numpy as np
import torch
import torch.nn as nn
from dp2 import utils
from .projector import F_RandomProj
from ..diffaug import diff_augment
from torch.nn.utils import spectral_norm


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def down_block(in_planes, out_planes):
    return nn.Sequential(
        conv2d(in_planes, out_planes, 4, 2, 1),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )


def down_block_patch(in_planes, out_planes):
    return nn.Sequential(
        down_block(in_planes, out_planes),
        conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SingleDisc(nn.Module):
    def __init__(
        self, cin, start_sz, end_sz, patch,
        last_ksize
    ):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64,
                        128: 64, 256: 32, 512: 16, 1024: 8,
                        }

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz
        channel_dict[start_sz] = cin

        layers = []
        # Down Blocks
        DB = down_block_patch if patch else down_block
        while start_sz > end_sz:
            layers.append(DB(channel_dict[start_sz], channel_dict[start_sz // 2]))
            start_sz = start_sz // 2
        p = 0 if last_ksize == 4 else 1
        layers.append(conv2d(channel_dict[end_sz], 1, last_ksize, padding=p, bias=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MultiScaleD(nn.Module):
    def __init__(self, channels, resolutions, num_discs, patch, last_ksize):
        super().__init__()
        assert num_discs in [1, 2, 3, 4]
        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]

        self.mini_discs = nn.ModuleList()
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            self.mini_discs.append(
                SingleDisc(cin=cin, start_sz=start_sz, end_sz=8, patch=patch, last_ksize=last_ksize)
            )

    def forward(self, features):
        all_logits = []
        for feat, disc in zip(features, self.mini_discs):
            all_logits.append(disc(feat))
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        backbones: List[dict],
        num_discs,
        diffaug_policy,
        patch,
        last_ksize,
    ):
        super().__init__()
        self.feature_networks = nn.ModuleList()
        self.discriminators = nn.ModuleList()
        for b in backbones:
            feature_network = F_RandomProj(**b)
            if feature_network.jit_script:
                feature_network = torch.jit.script(feature_network)
            self.feature_networks.append(feature_network)
            in_channels = feature_network.CHANNELS
            discriminator = MultiScaleD(
                channels=in_channels,
                resolutions=feature_network.RESOLUTIONS,
                num_discs=num_discs,
                patch=patch,
                last_ksize=last_ksize,
            )
            self.discriminators.append(discriminator)
        utils.set_requires_grad(self.feature_networks, False)
        utils.set_requires_grad(self.discriminators, True)
        self.diffaug_policy = diffaug_policy

    def train(self, mode=True):
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, img: torch.Tensor,  fnet_grad=True, **kwargs):
        img = img.add(1).div(2)

        logits = []
        for backbone, D in zip(self.feature_networks, self.discriminators):
            # Discriminator forward pass does not need gradient for feature network.
            with torch.set_grad_enabled(fnet_grad):
                x = img.sub(backbone.mean).div(backbone.std)
                x, _ = diff_augment(x, None, self.diffaug_policy)
                features = backbone(x)

            logits.append(D(features))
        return dict(logits=logits)
