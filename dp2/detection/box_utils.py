import numpy as np


def expand_bbox_to_ratio(bbox, imshape, target_aspect_ratio):
    x0, y0, x1, y1 = [int(_) for _ in bbox]
    h, w = y1 - y0, x1 - x0
    cur_ratio = h / w

    if cur_ratio == target_aspect_ratio:
        return [x0, y0, x1, y1]
    if cur_ratio < target_aspect_ratio:
        target_height = int(w*target_aspect_ratio)
        y0, y1 = expand_axis(y0, y1, target_height, imshape[0])
    else:
        target_width = int(h/target_aspect_ratio)
        x0, x1 = expand_axis(x0, x1, target_width, imshape[1])
    return x0, y0, x1, y1


def expand_axis(start, end, target_width, limit):
    # Can return a bbox outside of limit
    cur_width = end - start
    start = start - (target_width-cur_width)//2
    end = end + (target_width-cur_width)//2
    if end - start != target_width:
        end += 1
    assert end - start == target_width
    if start < 0 and end > limit:
        return start, end
    if start < 0 and end < limit:
        to_shift = min(0 - start, limit - end)
        start += to_shift
        end += to_shift
    if end > limit and start > 0:
        to_shift = min(end - limit, start)
        end -= to_shift
        start -= to_shift
    assert end - start == target_width
    return start, end


def expand_box(bbox, imshape, mask, percentage_background: float):
    assert isinstance(bbox[0], int)
    assert 0 < percentage_background < 1
    # Percentage in S
    mask_pixels = mask.long().sum().cpu()
    total_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    percentage_mask = mask_pixels / total_pixels
    if (1 - percentage_mask) > percentage_background:
        return bbox
    target_pixels = mask_pixels / (1 - percentage_background)
    x0, y0, x1, y1 = bbox
    H = y1 - y0
    W = x1 - x0
    p = np.sqrt(target_pixels/(H*W))
    target_width = int(np.ceil(p * W))
    target_height = int(np.ceil(p * H))
    x0, x1 = expand_axis(x0, x1, target_width, imshape[1])
    y0, y1 = expand_axis(y0, y1, target_height, imshape[0])
    return [x0, y0, x1, y1]


def expand_axises_by_percentage(bbox_XYXY, imshape, percentage):
    x0, y0, x1, y1 = bbox_XYXY
    H = y1 - y0
    W = x1 - x0
    expansion = int(((H*W)**0.5) * percentage)
    new_width = W + expansion
    new_height = H + expansion
    x0, x1 = expand_axis(x0, x1, min(new_width, imshape[1]), imshape[1])
    y0, y1 = expand_axis(y0, y1, min(new_height, imshape[0]), imshape[0])
    return [x0, y0, x1, y1]


def get_expanded_bbox(
        bbox_XYXY,
        imshape,
        mask,
        percentage_background: float,
        axis_minimum_expansion: float,
        target_aspect_ratio: float):
    bbox_XYXY = bbox_XYXY.long().cpu().numpy().tolist()
    # Expand each axis of the bounding box by a minimum percentage
    bbox_XYXY = expand_axises_by_percentage(bbox_XYXY, imshape, axis_minimum_expansion)
    # Find the minimum bbox with the aspect ratio. Can be outside of imshape
    bbox_XYXY = expand_bbox_to_ratio(bbox_XYXY, imshape, target_aspect_ratio)
    # Expands square box such that X% of the bbox is background
    bbox_XYXY = expand_box(bbox_XYXY, imshape, mask, percentage_background)
    assert isinstance(bbox_XYXY[0], (int, np.int64))
    return bbox_XYXY


def include_box(bbox, minimum_area, aspect_ratio_range, min_bbox_ratio_inside, imshape):
    def area_inside_ratio(bbox, imshape):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_inside = (min(bbox[2], imshape[1]) - max(0, bbox[0])) * (min(imshape[0], bbox[3]) - max(0, bbox[1]))
        return area_inside / area
    ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    if area_inside_ratio(bbox, imshape) < min_bbox_ratio_inside:
        return False
    if ratio <= aspect_ratio_range[0] or ratio >= aspect_ratio_range[1] or area < minimum_area:
        return False
    return True
