import torch
import tops


def denormalize_img(image, mean=0.5, std=0.5):
    image = image * std + mean
    image = torch.clamp(image.float(), 0, 1)
    image = (image * 255)
    image = torch.round(image)
    return image / 255


@torch.no_grad()
def im2numpy(images, to_uint8=False, denormalize=False):
    if denormalize:
        images = denormalize_img(images)
        if images.dtype != torch.uint8:
            images = images.clamp(0, 1)
    return tops.im2numpy(images, to_uint8=to_uint8)


@torch.no_grad()
def im2torch(im, cuda=False, normalize=True, to_float=True):
    im = tops.im2torch(im, cuda=cuda, to_float=to_float)
    if normalize:
        assert im.min() >= 0.0 and im.max() <= 1.0
        if normalize:
            im = im * 2 - 1
    return im


@torch.no_grad()
def binary_dilation(im: torch.Tensor, kernel: torch.Tensor):
    assert len(im.shape) == 4
    assert len(kernel.shape) == 2
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = kernel.shape[-1]//2
    assert kernel.shape[-1] % 2 != 0
    if isinstance(im, torch.cuda.FloatTensor):
        im, kernel = im.half(), kernel.half()
    else:
        im, kernel = im.float(), kernel.float()
    im = torch.nn.functional.conv2d(
        im, kernel, groups=im.shape[1], padding=padding)
    im = im > 0.5
    return im


@torch.no_grad()
def binary_erosion(im: torch.Tensor, kernel: torch.Tensor):
    assert len(im.shape) == 4
    assert len(kernel.shape) == 2
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = kernel.shape[-1]//2
    assert kernel.shape[-1] % 2 != 0
    if isinstance(im, torch.cuda.FloatTensor):
        im, kernel = im.half(), kernel.half()
    else:
        im, kernel = im.float(), kernel.float()
    ksum = kernel.sum()
    padding = (padding, padding, padding, padding)
    im = torch.nn.functional.pad(im, padding, mode="reflect")
    im = torch.nn.functional.conv2d(
        im, kernel, groups=im.shape[1])
    return im.round() == ksum


def set_requires_grad(value: torch.nn.Module, requires_grad: bool):
    if isinstance(value, (list, tuple)):
        for param in value:
            param.requires_grad = requires_grad
        return
    for p in value.parameters():
        p.requires_grad = requires_grad


def forward_D_fake(batch, fake_img, discriminator, **kwargs):
    fake_batch = {k: v for k, v in batch.items() if k != "img"}
    fake_batch["img"] = fake_img
    return discriminator(**fake_batch, **kwargs)


def remove_pad(x: torch.Tensor, bbox_XYXY, imshape):
    """
    Remove padding that is shown as negative 
    """
    H, W = imshape
    x0, y0, x1, y1 = bbox_XYXY
    padding = [
        max(0, -x0),
        max(0, -y0),
        max(x1 - W, 0),
        max(y1 - H, 0)
    ]
    x0, y0 = padding[:2]
    x1 = x.shape[2] - padding[2]
    y1 = x.shape[1] - padding[3]
    return x[:, y0:y1, x0:x1]


def crop_box(x: torch.Tensor, bbox_XYXY) -> torch.Tensor:
    """
        Crops x by bbox_XYXY. 
    """
    x0, y0, x1, y1 = bbox_XYXY
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, x.shape[-1])
    y1 = min(y1, x.shape[-2])
    return x[..., y0:y1, x0:x1]


def torch_wasserstein_loss(tensor_a, tensor_b):
    # Compute the first Wasserstein distance between two 1D distributions.
    return (torch_cdf_loss(tensor_a, tensor_b, p=1))


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b), 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b), p), dim=-1), 1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss
