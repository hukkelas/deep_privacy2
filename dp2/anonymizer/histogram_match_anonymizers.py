
import torch
import tops
import numpy as np
from kornia.color import rgb_to_hsv
from dp2 import utils
from kornia.enhance import histogram
from .anonymizer import Anonymizer
import torchvision.transforms.functional as F
from skimage.exposure import match_histograms
from kornia.filters import gaussian_blur2d


class LatentHistogramMatchAnonymizer(Anonymizer):

    def forward_G(
        self,
        G,
        batch,
        multi_modal_truncation: bool,
        amp: bool,
        z_idx: int,
        truncation_value: float,
        idx: int,
        n_sampling_steps: int = 1,
        all_styles=None,
    ):
        batch["img"] = F.normalize(batch["img"].float(), [0.5*255, 0.5*255, 0.5*255], [0.5*255, 0.5*255, 0.5*255])
        batch["condition"] = batch["mask"].float() * batch["img"]

        assert z_idx is None and all_styles is None, "Arguments not supported with n_sampling_steps > 1."
        real_hls = rgb_to_hsv(utils.denormalize_img(batch["img"]))
        real_hls[:, 0] /= 2 * torch.pi
        indices = [1, 2]
        hist_kwargs = dict(
            bins=torch.linspace(0, 1, 256, dtype=torch.float32, device=tops.get_device()),
            bandwidth=torch.tensor(1., device=tops.get_device()))
        real_hist = [histogram(real_hls[:, i].flatten(start_dim=1), **hist_kwargs) for i in indices]
        if multi_modal_truncation:
            w = G.style_net.multi_modal_truncate(
                truncation_value=truncation_value, n=batch["condition"].shape[0]).detach()
        else:
            w = G.style_net.get_truncated(truncation_value, n=batch["condition"].shape[0]).detach()
        assert z_idx is None and all_styles is None, "Arguments not supported with n_sampling_steps > 1."
        w.requires_grad = True
        optim = torch.optim.Adam([w])
        for j in range(n_sampling_steps):
            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast(amp):
                    anonymized_im = G(**batch, truncation_value=None, w=w)["img"]
                fake_hls = rgb_to_hsv(anonymized_im*0.5 + 0.5)
                fake_hls[:, 0] /= 2 * torch.pi
                fake_hist = [histogram(fake_hls[:, i].flatten(start_dim=1), **hist_kwargs) for i in indices]
                dist = sum([utils.torch_wasserstein_loss(r, f) for r, f in zip(real_hist, fake_hist)])
                dist.backward()
                if w.grad.sum() == 0:
                    break
                assert w.grad.sum() != 0
                optim.step()
                optim.zero_grad()
                if dist < 0.02:
                    break
        anonymized_im = (anonymized_im+1).div(2).clamp(0, 1).mul(255)
        return anonymized_im


class HistogramMatchAnonymizer(Anonymizer):

    def forward_G(self, G, batch, *args, **kwargs):
        rimg = batch["img"]
        anonymized_im = super().forward_G(G, batch, *args, **kwargs)

        equalized_gim = match_histograms(tops.im2numpy(anonymized_im.round().clamp(0, 255).byte()), tops.im2numpy(rimg))
        if equalized_gim.dtype != np.uint8:
            equalized_gim = equalized_gim.astype(np.float32)
            assert equalized_gim.dtype == np.float32, equalized_gim.dtype
            equalized_gim = tops.im2torch(equalized_gim, to_float=False)[0]
        else:
            equalized_gim = tops.im2torch(equalized_gim, to_float=False).float()[0]
        equalized_gim = equalized_gim.to(device=rimg.device)
        assert equalized_gim.dtype == torch.float32
        gaussian_mask = 1 - (batch["maskrcnn_mask"][0].repeat(3, 1, 1) > 0.5).float()

        gaussian_mask = gaussian_blur2d(gaussian_mask[None], kernel_size=[19, 19], sigma=[10, 10])[0]
        gaussian_mask = gaussian_mask / gaussian_mask.max()
        anonymized_im = gaussian_mask * equalized_gim + (1-gaussian_mask) * anonymized_im
        return anonymized_im
