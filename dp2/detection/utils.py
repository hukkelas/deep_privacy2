import cv2
import numpy as np
import torch
import tops
from skimage.morphology import disk
from torchvision.transforms.functional import resize, InterpolationMode
from functools import lru_cache


@lru_cache(maxsize=200)
def get_kernel(n: int):
    kernel = disk(n, dtype=bool)
    return tops.to_cuda(torch.from_numpy(kernel).bool())


def transform_embedding(E: torch.Tensor, S: torch.Tensor, exp_bbox, E_bbox, target_imshape):
    """
        Transforms the detected embedding/mask directly to the target image shape
    """

    C, HE, WE = E.shape
    assert E_bbox[0] >= exp_bbox[0], (E_bbox, exp_bbox)
    assert E_bbox[2] >= exp_bbox[0]
    assert E_bbox[1] >= exp_bbox[1]
    assert E_bbox[3] >= exp_bbox[1]
    assert E_bbox[2] <= exp_bbox[2]
    assert E_bbox[3] <= exp_bbox[3]

    x0 = int(np.round((E_bbox[0] - exp_bbox[0]) / (exp_bbox[2] - exp_bbox[0]) * target_imshape[1]))
    x1 = int(np.round((E_bbox[2] - exp_bbox[0]) / (exp_bbox[2] - exp_bbox[0]) * target_imshape[1]))
    y0 = int(np.round((E_bbox[1] - exp_bbox[1]) / (exp_bbox[3] - exp_bbox[1]) * target_imshape[0]))
    y1 = int(np.round((E_bbox[3] - exp_bbox[1]) / (exp_bbox[3] - exp_bbox[1]) * target_imshape[0]))
    new_E = torch.zeros((C, *target_imshape), device=E.device, dtype=torch.float32)
    new_S = torch.zeros((target_imshape), device=S.device, dtype=torch.bool)

    E = resize(E, (y1-y0, x1-x0), antialias=True, interpolation=InterpolationMode.BILINEAR)
    new_E[:, y0:y1, x0:x1] = E
    S = resize(S[None].float(), (y1-y0, x1-x0), antialias=True, interpolation=InterpolationMode.BILINEAR)[0] > 0
    new_S[y0:y1, x0:x1] = S
    return new_E, new_S


def pairwise_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor):
    """
        mask: shape [N, H, W]
    """
    assert len(mask1.shape) == 3
    assert len(mask2.shape) == 3
    assert mask1.device == mask2.device, (mask1.device, mask2.device)
    assert mask2.dtype == mask2.dtype
    assert mask1.dtype == torch.bool
    assert mask1.shape[1:] == mask2.shape[1:]
    N1, H1, W1 = mask1.shape
    N2, H2, W2 = mask2.shape
    iou = torch.zeros((N1, N2), dtype=torch.float32)
    for i in range(N1):
        cur = mask1[i:i+1]
        inter = torch.logical_and(cur, mask2).flatten(start_dim=1).float().sum(dim=1).cpu()
        union = torch.logical_or(cur, mask2).flatten(start_dim=1).float().sum(dim=1).cpu()
        iou[i] = inter / union
    return iou


def find_best_matches(mask1: torch.Tensor, mask2: torch.Tensor, iou_threshold: float):
    N1 = mask1.shape[0]
    N2 = mask2.shape[0]
    ious = pairwise_mask_iou(mask1, mask2).cpu().numpy()
    indices = np.array([idx for idx, iou in np.ndenumerate(ious)])
    ious = ious.flatten()
    mask = ious >= iou_threshold
    ious = ious[mask]
    indices = indices[mask]

    # do not sort by iou to keep ordering of mask rcnn / cse sorting.
    taken1 = np.zeros((N1), dtype=bool)
    taken2 = np.zeros((N2), dtype=bool)
    matches = []
    for i, j in indices:
        if taken1[i].any() or taken2[j].any():
            continue
        matches.append((i, j))
        taken1[i] = True
        taken2[j] = True
    return matches


def combine_cse_maskrcnn_dets(segmentation: torch.Tensor, cse_dets: dict, iou_threshold: float):
    assert 0 < iou_threshold <= 1
    matches = find_best_matches(segmentation, cse_dets["im_segmentation"], iou_threshold)
    H, W = segmentation.shape[1:]
    new_seg = torch.zeros((len(matches), H, W), dtype=torch.bool, device=segmentation.device)
    cse_im_seg = cse_dets["im_segmentation"]
    for idx, (i, j) in enumerate(matches):
        new_seg[idx] = torch.logical_or(segmentation[i], cse_im_seg[j])
    cse_dets = dict(
        instance_segmentation=cse_dets["instance_segmentation"][[j for (i, j) in matches]],
        instance_embedding=cse_dets["instance_embedding"][[j for (i, j) in matches]],
        bbox_XYXY=cse_dets["bbox_XYXY"][[j for (i, j) in matches]],
        scores=cse_dets["scores"][[j for (i, j) in matches]],
    )
    return new_seg, cse_dets, np.array(matches).reshape(-1, 2)


def initialize_cse_boxes(segmentation: torch.Tensor, cse_boxes: torch.Tensor):
    """
        cse_boxes can be outside of segmentation.
    """
    boxes = masks_to_boxes(segmentation)

    assert boxes.shape == cse_boxes.shape, (boxes.shape, cse_boxes.shape)
    combined = torch.stack((boxes, cse_boxes), dim=-1)
    boxes = torch.cat((
        combined[:, :2].min(dim=2).values,
        combined[:, 2:].max(dim=2).values,
    ), dim=1)
    return boxes


def cut_pad_resize(x: torch.Tensor, bbox, target_shape, fdf_resize=False):
    """
        Crops or pads x to fit in the bbox and resize to target shape.
    """
    C, H, W = x.shape
    x0, y0, x1, y1 = bbox

    if y0 > 0 and x0 > 0 and x1 <= W and y1 <= H:
        new_x = x[:, y0:y1, x0:x1]
    else:
        new_x = torch.zeros(((C, y1-y0, x1-x0)), dtype=x.dtype, device=x.device)
        y0_t = max(0, -y0)
        y1_t = min(y1-y0, (y1-y0)-(y1-H))
        x0_t = max(0, -x0)
        x1_t = min(x1-x0, (x1-x0)-(x1-W))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, W)
        y1 = min(y1, H)
        new_x[:, y0_t:y1_t, x0_t:x1_t] = x[:, y0:y1, x0:x1]
    # Nearest upsampling often generates more sharp synthesized identities.
    interp = InterpolationMode.BICUBIC
    if (y1-y0) < target_shape[0] and (x1-x0) < target_shape[1]:
        interp = InterpolationMode.NEAREST
    antialias = interp == InterpolationMode.BICUBIC
    if x1 - x0 == target_shape[1] and y1 - y0 == target_shape[0]:
        return new_x
    if x.dtype == torch.bool:
        new_x = resize(new_x.float(), target_shape, interpolation=InterpolationMode.NEAREST) > 0.5
    elif x.dtype == torch.float32:
        new_x = resize(new_x, target_shape, interpolation=interp, antialias=antialias)
    elif x.dtype == torch.uint8:
        if fdf_resize:  # FDF dataset is created with cv2 INTER_AREA.
            # Incorrect resizing generates noticeable poorer inpaintings.
            upsampling = ((y1-y0) * (x1-x0)) < (target_shape[0] * target_shape[1])
            if upsampling:
                new_x = resize(new_x.float(), target_shape, interpolation=InterpolationMode.BICUBIC,
                               antialias=True).round().clamp(0, 255).byte()
            else:
                device = new_x.device
                new_x = new_x.permute(1, 2, 0).cpu().numpy()
                new_x = cv2.resize(new_x, target_shape[::-1], interpolation=cv2.INTER_AREA)
                new_x = torch.from_numpy(np.rollaxis(new_x, 2)).to(device)
        else:
            new_x = resize(new_x.float(), target_shape, interpolation=interp,
                           antialias=antialias).round().clamp(0, 255).byte()
    else:
        raise ValueError(f"Not supported dtype: {x.dtype}")
    return new_x


def masks_to_boxes(segmentation: torch.Tensor):
    assert len(segmentation.shape) == 3
    x = segmentation.any(dim=1).byte()  # Compress rows
    x0 = x.argmax(dim=1)

    x1 = segmentation.shape[2] - x.flip(dims=(1,)).argmax(dim=1)
    y = segmentation.any(dim=2).byte()
    y0 = y.argmax(dim=1)
    y1 = segmentation.shape[1] - y.flip(dims=(1,)).argmax(dim=1)
    return torch.stack([x0, y0, x1, y1], dim=1)
