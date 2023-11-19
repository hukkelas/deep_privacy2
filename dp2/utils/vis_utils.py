import torch
import tops
import cv2
import torchvision.transforms.functional as F
from typing import Optional, List, Union, Tuple
from .cse import from_E_to_vertex
import numpy as np
from tops import download_file
from .torch_utils import (
    denormalize_img, binary_dilation, binary_erosion,
    remove_pad, crop_box)
from torchvision.utils import _generate_color_palette
from PIL import Image, ImageColor, ImageDraw


def get_coco_keypoints():
    # From: https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/keypoints.py
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    connectivity = {
        "nose": "left_eye",
        "left_eye": "right_eye",
        "right_eye": "nose",
        "left_ear": "left_eye",
        "right_ear": "right_eye",
        "left_shoulder": "nose",
        "right_shoulder": "nose",
        "left_elbow": "left_shoulder",
        "right_elbow": "right_shoulder",
        "left_wrist": "left_elbow",
        "right_wrist": "right_elbow",
        "left_hip": "left_shoulder",
        "right_hip": "right_shoulder",
        "left_knee": "left_hip",
        "right_knee": "right_hip",
        "left_ankle": "left_knee",
        "right_ankle": "right_knee"
    }
    connectivity_indices = [
        (sidx, keypoints.index(connectivity[kp]))
        for sidx, kp in enumerate(keypoints)
    ]
    return keypoints, keypoint_flip_map, connectivity_indices


def get_coco_colors():
    return [
        *["red"]*5,
        "blue",
        "green",
        "blue",
        "green",
        "blue",
        "green",
        "purple",
        "orange",
        "purple",
        "orange",
        "purple",
        "orange",
    ]


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    visible: Optional[List[List[bool]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = None,
    width: int = None,
) -> torch.Tensor:
    """
    Function taken from torchvision source code.  Added in torchvision 0.12

    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")
    if width is None:
        width = int(max(max(image.shape[-2:]) * 0.01, 1))
    if radius is None:
        radius = int(max(max(image.shape[-2:]) * 0.01, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    if isinstance(keypoints, torch.Tensor):
        img_kpts = keypoints.to(torch.int64).tolist()
    else:
        assert isinstance(keypoints, np.ndarray)
        img_kpts = keypoints.astype(int).tolist()
    colors = get_coco_colors()
    for inst_id, kpt_inst in enumerate(img_kpts):

        for kpt_id, kpt in enumerate(kpt_inst):
            if visible is not None and int(visible[inst_id][kpt_id]) == 0:
                continue
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius

            draw.ellipse([x1, y1, x2, y2], fill=colors[kpt_id], outline=None, width=0)

        if connectivity is not None:
            for connection in connectivity:
                if connection[1] >= len(kpt_inst) or connection[0] >= len(kpt_inst):
                    continue
                if visible is not None and (int(visible[inst_id][connection[1]]) == 0 or int(visible[inst_id][connection[0]]) == 0):
                    continue

                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width, fill=colors[connection[1]]
                )

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def visualize_keypoints(img, keypoints):
    img = img.clone()
    keypoints = keypoints.clone()
    keypoints[:, :, 0] *= img.shape[-1]
    keypoints[:, :, 1] *= img.shape[-2]
    _, _, connectivity = get_coco_keypoints()
    connectivity = np.array(connectivity)
    visible = None
    if keypoints.shape[-1] == 3:
        visible = keypoints[:, :, 2] > 0
    for idx in range(img.shape[0]):
        img[idx] = draw_keypoints(
            img[idx], keypoints[idx:idx+1].long(), colors="red",
            connectivity=connectivity, visible=visible[idx:idx+1])
    return img


def visualize_batch(
        img: torch.Tensor, mask: torch.Tensor,
        vertices: torch.Tensor = None,
        E_mask: torch.Tensor = None,
        embed_map: torch.Tensor = None,
        semantic_mask: torch.Tensor = None,
        embedding: torch.Tensor = None,
        keypoints: torch.Tensor = None,
        maskrcnn_mask: torch.Tensor = None,
        **kwargs) -> torch.ByteTensor:
    img = denormalize_img(img).mul(255).round().clamp(0, 255).byte()
    img = draw_mask(img, mask)
    if maskrcnn_mask is not None and maskrcnn_mask.shape == mask.shape:
        img = draw_mask(img, maskrcnn_mask)
    if vertices is not None or embedding is not None:
        assert E_mask is not None
        assert embed_map is not None
        img, E_mask, embedding, embed_map, vertices = tops.to_cpu([
            img, E_mask, embedding, embed_map, vertices
        ])
        img = draw_cse(img, E_mask, embedding, embed_map, vertices)
    elif semantic_mask is not None:
        img = draw_segmentation_masks(img, semantic_mask)
    if keypoints is not None:
        img = visualize_keypoints(img, keypoints)
    return img


@torch.no_grad()
def draw_cse(
        img: torch.Tensor, E_seg: torch.Tensor,
        embedding: torch.Tensor = None,
        embed_map: torch.Tensor = None,
        vertices: torch.Tensor = None, t=0.7
):
    """
        E_seg: 1 for areas with embedding
    """
    assert img.dtype == torch.uint8
    img = img.view(-1, *img.shape[-3:])
    E_seg = E_seg.view(-1, 1, *E_seg.shape[-2:])
    if vertices is None:
        assert embedding is not None
        assert embed_map is not None
        embedding = embedding.view(-1, *embedding.shape[-3:])
        vertices = torch.stack(
            [from_E_to_vertex(e[None], e_seg[None].logical_not().float(), embed_map)
             for e, e_seg in zip(embedding, E_seg)])

    i = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)
    colormap_JET = torch.from_numpy(cv2.applyColorMap(i, cv2.COLORMAP_JET)[0])
    color_embed_map, _ = np.load(download_file(
        "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"), allow_pickle=True)
    color_embed_map = torch.from_numpy(color_embed_map).float()[:, 0]
    color_embed_map -= color_embed_map.min()
    color_embed_map /= color_embed_map.max()
    vertx2idx = (color_embed_map*255).long()
    vertx2colormap = colormap_JET[vertx2idx]

    vertices = vertices.view(-1, *vertices.shape[-2:])
    E_seg = E_seg.view(-1, 1, *E_seg.shape[-2:])
    # This operation might be good to do on cpu...

    E_color = vertx2colormap[vertices.long()]
    E_color = E_color.to(E_seg.device)
    E_color = E_color.permute(0, 3, 1, 2)
    E_color = E_color*E_seg.byte()

    m = E_seg.bool().repeat(1, 3, 1, 1)
    img[m] = (img[m] * (1-t) + t * E_color[m]).byte()
    return img


def draw_cse_all(
        embedding: List[torch.Tensor], E_mask: List[torch.Tensor],
        im: torch.Tensor, boxes_XYXY: list, embed_map: torch.Tensor, t=0.7):
    """
        E_seg: 1 for areas with embedding
    """
    assert len(im.shape) == 3, im.shape
    assert im.dtype == torch.uint8

    N = len(E_mask)
    im = im.clone()
    for i in range(N):
        assert len(E_mask[i].shape) == 2
        assert len(embedding[i].shape) == 3
        assert embed_map.shape[1] == embedding[i].shape[0]
        assert len(boxes_XYXY[i]) == 4
        E = embedding[i]
        x0, y0, x1, y1 = boxes_XYXY[i]
        E = F.resize(E, (y1-y0, x1-x0), antialias=True)
        s = E_mask[i].float()
        s = (F.resize(s.squeeze()[None], (y1-y0, x1-x0), antialias=True) > 0).float()
        box = boxes_XYXY[i]

        im_ = crop_box(im, box)
        s = remove_pad(s, box, im.shape[1:])
        E = remove_pad(E, box, im.shape[1:])
        E_color = draw_cse(img=im_, E_seg=s[None], embedding=E[None], embed_map=embed_map)[0]
        E_color = E_color.to(im.device)
        s = s.bool().repeat(3, 1, 1)
        crop_box(im, box)[s] = (im_[s] * (1-t) + t * E_color[s]).byte()
    return im


@torch.no_grad()
def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (list or None): List containing the colors of the masks. The colors can
            be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            When ``masks`` has a single entry of shape (H, W), you can pass a single color instead of a list
            with one element. By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")
    num_masks = masks.size()[0]
    if num_masks == 0:
        return image
    if colors is None:
        colors = _generate_color_palette(num_masks)
    if not isinstance(colors[0], (Tuple, List)):
        colors = [colors for i in range(num_masks)]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        color = torch.tensor(color, dtype=out_dtype, device=masks.device)
        colors_.append(color)
    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


def draw_mask(im: torch.Tensor, mask: torch.Tensor, t=0.2, color=(255, 255, 255), visualize_instances=True):
    """
        Visualize mask where mask = 0.
        Supports multiple instances.
        mask shape: [N, C, H, W], where C is different instances in same image.
    """
    orig_imshape = im.shape
    if mask.numel() == 0:
        return im
    assert len(mask.shape) in (3, 4), mask.shape
    mask = mask.view(-1, *mask.shape[-3:])
    im = im.view(-1, *im.shape[-3:])
    assert im.dtype == torch.uint8, im.dtype
    assert 0 <= t <= 1
    if not visualize_instances:
        mask = mask.any(dim=1, keepdim=True)
    mask = mask.bool()
    kernel = torch.ones((3, 3), dtype=mask.dtype, device=mask.device)
    outer_border = binary_dilation(mask, kernel).logical_xor(mask)
    outer_border = outer_border.any(dim=1, keepdim=True).repeat(1, 3, 1, 1) > 0
    inner_border = binary_erosion(mask, kernel).logical_xor(mask)
    inner_border = inner_border.any(dim=1, keepdim=True).repeat(1, 3, 1, 1) > 0
    mask = (mask == 0).any(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    color = torch.tensor(color).to(im.device).byte().view(1, 3, 1, 1)  # .repeat(1, *im.shape[1:])
    color = color.repeat(im.shape[0], 1, *im.shape[-2:])
    im[mask] = (im[mask] * (1-t) + t * color[mask]).byte()
    im[outer_border] = 255
    im[inner_border] = 0
    return im.view(*orig_imshape)


def draw_cropped_masks(im: torch.Tensor, mask: torch.Tensor, boxes: torch.Tensor, **kwargs):
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = boxes[i]
        orig_shape = (y1-y0, x1-x0)
        m = F.resize(mask[i], orig_shape, F.InterpolationMode.NEAREST).squeeze()[None]
        m = remove_pad(m, boxes[i], im.shape[-2:])
        crop_box(im, boxes[i]).set_(draw_mask(crop_box(im, boxes[i]), m))
    return im


def draw_cropped_keypoints(im: torch.Tensor, all_keypoints: torch.Tensor, boxes: torch.Tensor, **kwargs):
    n_boxes = boxes.shape[0]
    tops.assert_shape(all_keypoints, (n_boxes, 17, 3))
    im = im.clone()
    for i, box in enumerate(boxes):

        x0, y0, x1, y1 = boxes[i]
        orig_shape = (y1-y0, x1-x0)
        keypoints = all_keypoints[i].clone()
        keypoints[:, 0] *= orig_shape[1]
        keypoints[:, 1] *= orig_shape[0]
        keypoints = keypoints.long()
        _, _, connectivity = get_coco_keypoints()
        connectivity = np.array(connectivity)
        visible = (keypoints[:, 2] > .5)
        # Remove padding from keypoints before visualization
        keypoints[:, 0] += min(x0, 0)
        keypoints[:, 1] += min(y0, 0)
        im_with_kp = draw_keypoints(
            crop_box(im, box), keypoints[None], colors="red", connectivity=connectivity, visible=visible[None])
        crop_box(im, box).copy_(im_with_kp)
    return im
