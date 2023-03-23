import functools
import torch
import tops
from tops import logger
from dp2.utils import forward_D_fake
from .utils import nsgan_d_loss, nsgan_g_loss
from .r1_regularization import r1_regularization
from .pl_regularization import PLRegularization


class StyleGAN2Loss:

    def __init__(
            self,
            D,
            G,
            r1_opts: dict,
            EP_lambd: float,
            lazy_reg_interval: int,
            lazy_regularization: bool,
            pl_reg_opts: dict,
        ) -> None:
        self.gradient_step_D = 0
        self._lazy_reg_interval = lazy_reg_interval
        self.D = D
        self.G = G
        self.EP_lambd = EP_lambd
        self.lazy_regularization = lazy_regularization
        self.r1_reg = functools.partial(
            r1_regularization, **r1_opts, lazy_reg_interval=lazy_reg_interval,
            lazy_regularization=lazy_regularization)
        self.do_PL_Reg = False
        if pl_reg_opts.weight > 0:
            self.pl_reg = PLRegularization(**pl_reg_opts)
            self.do_PL_Reg = True
            self.pl_start_nimg = pl_reg_opts.start_nimg

    def D_loss(self, batch: dict, grad_scaler):
        to_log = {}
        # Forward through G and D
        do_GP = self.lazy_regularization and self.gradient_step_D % self._lazy_reg_interval == 0
        if do_GP:
            batch["img"] = batch["img"].detach().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            with torch.no_grad():
                G_fake = self.G(**batch, update_emas=True)
            D_out_real = self.D(**batch)

            D_out_fake = forward_D_fake(batch, G_fake["img"], self.D)

            # Non saturating loss
            nsgan_loss = nsgan_d_loss(D_out_real["score"], D_out_fake["score"])
            tops.assert_shape(nsgan_loss, (batch["img"].shape[0], ))
            to_log["d_loss"] = nsgan_loss.mean()
            total_loss = nsgan_loss
            epsilon_penalty = D_out_real["score"].pow(2).view(-1)
            to_log["epsilon_penalty"] = epsilon_penalty.mean()
            tops.assert_shape(epsilon_penalty, total_loss.shape)
            total_loss = total_loss + epsilon_penalty * self.EP_lambd

        # Improved gradient penalty with lazy regularization
        # Gradient penalty applies specialized autocast.
        if do_GP:
            gradient_pen, grad_unscaled = self.r1_reg(
                batch["img"], D_out_real["score"], batch["mask"], scaler=grad_scaler)
            to_log["r1_gradient_penalty"] = grad_unscaled.mean()
            tops.assert_shape(gradient_pen, total_loss.shape)
            total_loss = total_loss + gradient_pen

        batch["img"] = batch["img"].detach().requires_grad_(False)
        if "score" in D_out_real:
            to_log["real_scores"] = D_out_real["score"]
            to_log["real_logits_sign"] = D_out_real["score"].sign()
            to_log["fake_logits_sign"] = D_out_fake["score"].sign()
            to_log["fake_scores"] = D_out_fake["score"]
        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        self.gradient_step_D += 1
        return total_loss.mean(), to_log

    def G_loss(self, batch: dict, grad_scaler):
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            to_log = {}
            # Forward through G and D
            G_fake = self.G(**batch)
            D_out_fake = forward_D_fake(batch, G_fake["img"], self.D)
            # Adversarial Loss
            total_loss = nsgan_g_loss(D_out_fake["score"]).view(-1)
            to_log["g_loss"] = total_loss.mean()
            tops.assert_shape(total_loss, (batch["img"].shape[0], ))

        if self.do_PL_Reg and logger.global_step() >= self.pl_start_nimg:
            pl_reg, to_log_ = self.pl_reg(self.G, batch, grad_scaler=grad_scaler)
            total_loss = total_loss + pl_reg.mean()
            to_log.update(to_log_)
        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        return total_loss.mean(), to_log
