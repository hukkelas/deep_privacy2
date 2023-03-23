import torch
import tops
import numpy as np
from sg3_torch_utils.ops import conv2d_gradfix

pl_mean_total = torch.zeros([])


class PLRegularization:

    def __init__(self, weight: float, batch_shrink: int, pl_decay: float, scale_by_mask: bool, **kwargs):
        self.pl_mean = torch.zeros([], device=tops.get_device())
        self.pl_weight = weight
        self.batch_shrink = batch_shrink
        self.pl_decay = pl_decay
        self.scale_by_mask = scale_by_mask

    def __call__(self, G, batch, grad_scaler):
        batch_size = batch["img"].shape[0] // self.batch_shrink
        batch = {k: v[:batch_size] for k, v in batch.items() if k != "embed_map"}
        if "embed_map" in batch:
            batch["embed_map"] = batch["embed_map"]
        z = G.get_z(batch["img"])

        with torch.cuda.amp.autocast(tops.AMP()):
            gen_ws = G.style_net(z)
            gen_img = G(**batch, w=gen_ws)["img"].float()
        pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        with conv2d_gradfix.no_weight_gradients():
            # Sums over HWC
            pl_grads = torch.autograd.grad(
                outputs=[grad_scaler.scale(gen_img * pl_noise)],
                inputs=[gen_ws],
                create_graph=True,
                grad_outputs=torch.ones_like(gen_img),
                only_inputs=True)[0]

        pl_grads = pl_grads.float() / grad_scaler.get_scale()
        if self.scale_by_mask:
            # Percentage of pixels known
            scaling = batch["mask"].flatten(start_dim=1).mean(dim=1).view(-1, 1)
            pl_grads = pl_grads / scaling
        pl_lengths = pl_grads.square().sum(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        if not torch.isnan(pl_mean).any():
            self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        to_log = dict(pl_penalty=pl_penalty.mean().detach())
        return pl_penalty.view(-1) * self.pl_weight, to_log
