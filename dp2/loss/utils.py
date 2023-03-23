import torch
import torch.nn.functional as F


def nsgan_g_loss(fake_score):
    """
        Non-saturating criterion from Goodfellow et al. 2014
    """
    return torch.nn.functional.softplus(-fake_score)


def nsgan_d_loss(real_score, fake_score):
    """
        Non-saturating criterion from Goodfellow et al. 2014
    """
    d_loss = F.softplus(-real_score) + F.softplus(fake_score)
    return d_loss.view(-1)


def smooth_masked_l1_loss(x, target, mask):
    """
        Pixel-wise l1 loss for the area indicated by mask
    """
    # Beta=.1 <-> square loss if pixel difference <= 12.8
    l1 = F.smooth_l1_loss(x*mask, target*mask, beta=.1, reduction="none").sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3])
    return l1
