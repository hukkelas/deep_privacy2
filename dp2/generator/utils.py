import torch
import tops
import torch
from torch.cuda.amp import custom_bwd, custom_fwd


@torch.no_grad()
def spatial_embed_keypoints(keypoints: torch.Tensor, x):
    tops.assert_shape(keypoints, (None, None, 3))
    B, N_K, _ = keypoints.shape
    H, W = x.shape[-2:]
    keypoint_spatial = torch.zeros(keypoints.shape[0], N_K, H, W, device=keypoints.device, dtype=torch.float32)
    x, y, visible = keypoints.chunk(3, dim=2)
    x = (x * W).round().long().clamp(0, W-1)
    y = (y * H).round().long().clamp(0, H-1)
    kp_idx = torch.arange(0, N_K, 1, device=keypoints.device, dtype=torch.long).view(1, -1, 1).repeat(B, 1, 1)
    pos = (kp_idx*(H*W) + y*W + x + 1)
    # Offset all by 1 to index invisible keypoints as 0
    pos = (pos * visible.round().long()).squeeze(dim=-1)
    keypoint_spatial = torch.zeros(keypoints.shape[0], N_K*H*W+1, device=keypoints.device, dtype=torch.float32)
    keypoint_spatial.scatter_(1, pos, 1)
    keypoint_spatial = keypoint_spatial[:, 1:].view(-1, N_K, H, W)
    return keypoint_spatial


class MaskOutput(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x_real, x_fake, mask):
        ctx.save_for_backward(mask)
        out = x_real * mask + (1-mask) * x_fake
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        fake_grad = grad_output
        mask, = ctx.saved_tensors
        fake_grad = fake_grad * (1 - mask)
        known_percentage = mask.view(mask.shape[0], -1).mean(dim=1)
        fake_grad = fake_grad / (1-known_percentage).view(-1, 1, 1, 1)
        return None, fake_grad, None


def mask_output(scale_grad, x_real, x_fake, mask):
    if scale_grad:
        return MaskOutput.apply(x_real, x_fake, mask)
    return x_real * mask + (1-mask) * x_fake
