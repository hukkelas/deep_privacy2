import torch
import tops


def r1_regularization(
        real_img, real_score, mask, lambd: float, lazy_reg_interval: int,
        lazy_regularization: bool,
        scaler: torch.cuda.amp.GradScaler, mask_out: bool,
        mask_out_scale: bool,
        **kwargs
):
    grad = torch.autograd.grad(
        outputs=scaler.scale(real_score),
        inputs=real_img,
        grad_outputs=torch.ones_like(real_score),
        create_graph=True,
        only_inputs=True,
    )[0]
    inv_scale = 1.0 / scaler.get_scale()
    grad = grad * inv_scale
    with torch.cuda.amp.autocast(tops.AMP()):
        if mask_out:
            grad = grad * (1 - mask)
        grad = grad.square().sum(dim=[1, 2, 3])
        if mask_out and mask_out_scale:
            total_pixels = real_img.shape[1] * real_img.shape[2] * real_img.shape[3]
            n_fake = (1-mask).sum(dim=[1, 2, 3])
            scaling = total_pixels / n_fake
            grad = grad * scaling
    if lazy_regularization:
        lambd_ = lambd * lazy_reg_interval / 2  # From stylegan2, lazy regularization
    return grad * lambd_, grad.detach()
