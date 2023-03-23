import torch
from .base import BaseGenerator
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F


class PixelationGenerator(BaseGenerator):

    def __init__(self, pixelation_size, **kwargs):
        super().__init__(z_channels=0)
        self.pixelation_size = pixelation_size
        self.z_channels = 0
        self.latent_space = None

    def forward(self, img, condition, mask, **kwargs):
        old_shape = img.shape[-2:]
        img = F.interpolate(img, size=(
            self.pixelation_size, self.pixelation_size), mode="bilinear", align_corners=True)
        img = F.interpolate(img, size=old_shape, mode="bilinear", align_corners=True)
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


class GaussianBlurGenerator(BaseGenerator):

    def __init__(self):
        super().__init__(z_channels=0)
        self.sigma = 7

    def forward(self, img, condition, mask, **kwargs):
        img_blur = gaussian_blur(img, kernel_size=min(self.sigma*3, img.shape[-1]), sigma=self.sigma)
        return dict(img=img * mask + (1-mask) * img_blur)
