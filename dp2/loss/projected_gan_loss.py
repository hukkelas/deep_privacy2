import numpy as np
import tops
import torch
from sg3_torch_utils.ops import upfirdn2d
from tops import logger


def blur(im, init_sigma, t, batch, **kwargs):
    batch = {k: v for k, v in batch.items()}
    sigma = max(t, 0) * init_sigma
    blur_size = int(np.round(sigma*3))
    if blur_size == 0:
        return im
    blur_f = torch.arange(-blur_size, blur_size + 1, device=im.device).div(sigma).square().neg().exp2()
    blur_f = blur_f / blur_f.sum()
    im = upfirdn2d.filter2d(im, blur_f)
    return im


class ProjectedGANLoss:

    def __init__(
            self,
            D,
            G,
            aug_fade_kimg,
            blur_init_sigma,
            loss_type="hinge",
        ) -> None:
        self.D = D
        self.G = G
        self.aug_fade_kimg = aug_fade_kimg
        self.do_PL_Reg = False
        self.aug_t = lambda: (1 - logger.global_step() / self.aug_fade_kimg)
        self.loss_type = loss_type
        assert self.loss_type in ["hinge", "masked-hinge"]
        self.blur_init_sigma = blur_init_sigma

    def forward_D(self, img, batch, fnet_grad: bool):
        if logger.global_step() < self.aug_fade_kimg:
            img = blur(im=img, batch=batch, t=self.aug_t(), init_sigma=self.blur_init_sigma)
        batch = {k: v for k, v in batch.items() if k != "img"}
        return self.D(**batch, img=img, fnet_grad=fnet_grad)

    def D_loss(self, batch: dict, **kwargs):
        to_log = {}
        # Forward through G and D
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            real_logits = self.forward_D(batch["img"], batch, False)["logits"]

            with torch.no_grad():
                fake_out = self.G(**batch, update_emas=True)
                fake_img = fake_out["img"]
            fake_logits = self.forward_D(fake_img, batch, False)["logits"]

            if self.loss_type == "hinge":
                fake_logits = [torch.cat([l.flatten(start_dim=1) for l in logits], dim=1) for logits in fake_logits]
                fake_loss = sum([(torch.ones_like(l) + l).relu().mean() for l in fake_logits])
                real_logits = [torch.cat([l.flatten(start_dim=1) for l in logits], dim=1) for logits in real_logits]
                real_loss = sum([(torch.ones_like(l) - l).relu().mean() for l in real_logits])

            if self.loss_type == "masked-hinge":
                real_loss, fake_loss, to_log_hinge = masked_hinge_D_loss(fake_logits,  real_logits, batch["mask"])
                # Reshape for logging
                fake_logits = [torch.cat([l.flatten(start_dim=1) for l in logits], dim=1) for logits in fake_logits]
                real_logits = [torch.cat([l.flatten(start_dim=1) for l in logits], dim=1) for logits in real_logits]

            real_logits = torch.cat([r.view(-1) for r in real_logits])
            fake_logits = torch.cat([r.view(-1) for r in fake_logits])

        total_loss = fake_loss + real_loss
        to_log = dict(
            real_scores=real_logits.mean(),
            fake_scores=fake_logits.mean(),
            real_loss=real_loss.mean(),
            fake_loss=fake_loss.mean(),
            fake_logits_sign=fake_logits.sign().mean(),
            real_logits_sign=real_logits.sign().mean()
        )
        if self.loss_type == "masked-hinge":
            to_log.update(to_log_hinge)
        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        return total_loss.mean(), to_log

    def G_loss(self, batch: dict, **kwargs):
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            to_log = {}
            # Forward through G and D
            G_fake = self.G(**batch)
            fake_logits = self.forward_D(G_fake["img"], batch, True)["logits"]
            fake_logits = [torch.cat([l.flatten(start_dim=1) for l in logits], dim=1) for logits in fake_logits]
            fake_loss = sum([(-l).mean() for l in fake_logits])

        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        return fake_loss.mean(), to_log


def masked_hinge_D_loss(fake_logits, real_logits, mask):
    fake_loss = 0
    real_loss = 0
    fake_logits_real_sign = 0
    fake_logits_fake_sign = 0
    for r_logits, f_logits in zip(real_logits, fake_logits):
        resized_fake_pixels = [
            torch.nn.functional.adaptive_max_pool2d(1 - mask, output_size=l.shape[-2:])
            for l in f_logits
        ]
        N_fake_pixels = sum([l.sum() for l in resized_fake_pixels])
        N_real_pixels = sum([np.prod(l.shape) for l in f_logits]) - N_fake_pixels

        for rl, fl, fake_pixels in zip(r_logits, f_logits, resized_fake_pixels):
            real_pixels = 1 - fake_pixels
            fake_loss = fake_loss + ((torch.ones_like(fl) + fl)*fake_pixels).relu().sum() / N_fake_pixels

            real_loss = real_loss + ((torch.ones_like(fl) - fl)*real_pixels).relu().sum() / N_real_pixels / 2
            real_loss = real_loss + ((torch.ones_like(fl) - rl)*real_pixels).relu().sum() / N_real_pixels / 2
            real_loss = real_loss + ((torch.ones_like(fl) - rl)*fake_pixels).relu().sum() / N_real_pixels
            fake_logits_real_sign += ((real_pixels * fl).sign().sum() / real_pixels.sum()).detach() / len(r_logits) / len(fake_logits)
            fake_logits_fake_sign += ((fake_pixels * fl).sign().sum() / fake_pixels.sum()).detach() / len(r_logits) / len(fake_logits)
    return real_loss, fake_loss, dict(
        fake_logits_real_sign=fake_logits_real_sign,
        fake_logits_fake_sign=fake_logits_fake_sign
    )
