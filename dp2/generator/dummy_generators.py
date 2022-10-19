import torch
import torch.nn as nn
from .base import BaseGenerator


class PixelationGenerator(BaseGenerator):

    def __init__(self, pixelation_size, **kwargs):
        super().__init__(z_channels=0)
        self.pixelation_size = pixelation_size
        self.z_channels = 0
        self.latent_space=None

    def forward(self, img, condition, mask, **kwargs):
        old_shape = img.shape[-2:]
        img = nn.functional.interpolate(img, size=(self.pixelation_size, self.pixelation_size), mode="bilinear", align_corners=True)
        img = nn.functional.interpolate(img, size=old_shape, mode="bilinear", align_corners=True)
        out = img*(1-mask) + condition*mask
        return {"img": out}


class MaskOutGenerator(BaseGenerator):

    def __init__(self, noise: str, **kwargs):
        super().__init__(z_channels=0)
        self.noise = noise
        self.z_channels = 0
        assert self.noise in ["rand", "constant"]
        self.latent_space = None

    def forward(self, img, condition, mask, **kwargs):
        
        if self.noise == "constant":
            img = torch.zeros_like(img)
        elif self.noise == "rand":
            img = torch.rand_like(img)
        out = img*(1-mask) + condition*mask
        return {"img": out}


class IdentityGenerator(BaseGenerator):

    def __init__(self):
        super().__init__(z_channels=0)

    def forward(self, img, condition, mask, **kwargs):
        return dict(img=img)