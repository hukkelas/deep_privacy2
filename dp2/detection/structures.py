import torch
import numpy as np
from dp2 import utils
from dp2.utils import vis_utils, crop_box
from .utils import (
    cut_pad_resize, masks_to_boxes,
    get_kernel, transform_embedding, initialize_cse_boxes
)
from .box_utils import get_expanded_bbox, include_box
import torchvision
import tops
from .box_utils_fdf import expand_bbox as expand_bbox_fdf
from dp2.data.transforms.transforms import InsertJointMap


class VehicleDetection:

    def __init__(self, segmentation: torch.BoolTensor) -> None:
        self.segmentation = segmentation
        self.boxes = masks_to_boxes(segmentation)
        assert self.boxes.shape[1] == 4, self.boxes.shape
        self.n_detections = self.segmentation.shape[0]
        area = (self.boxes[:, 3] - self.boxes[:, 1]) * (self.boxes[:, 2] - self.boxes[:, 0])

        sorted_idx = torch.argsort(area, descending=True)
        self.segmentation = self.segmentation[sorted_idx]
        self.boxes = self.boxes[sorted_idx].cpu()

    def pre_process(self):
        pass

    def get_crop(self, idx: int, im):
        assert idx < len(self)
        box = self.boxes[idx]
        im = crop_box(self.im, box)
        mask = crop_box(self.segmentation[idx])
        mask = mask == 0
        return dict(img=im, mask=mask.float(), boxes=box)

    def visualize(self, im):
        if len(self) == 0:
            return im
        im = vis_utils.draw_mask(im.clone(), self.segmentation.logical_not())
        return im

    def __len__(self):
        return self.n_detections

    @staticmethod
    def from_state_dict(state_dict, **kwargs):
        numel = np.prod(state_dict["shape"])
        arr = np.unpackbits(state_dict["segmentation"].numpy(), count=numel)
        segmentation = tops.to_cuda(torch.from_numpy(arr)).view(state_dict["shape"])
        return VehicleDetection(segmentation)

    def state_dict(self, **kwargs):
        segmentation = torch.from_numpy(np.packbits(self.segmentation.bool().cpu().numpy()))
        return dict(segmentation=segmentation, cls=self.__class__, shape=self.segmentation.shape)


class FaceDetection:

    def __init__(self,
                 boxes_ltrb: torch.LongTensor, target_imsize, fdf128_expand: bool,
                 keypoints: torch.Tensor = None,
                 **kwargs) -> None:

        self.boxes = boxes_ltrb.cpu()
        assert self.boxes.shape[1] == 4, self.boxes.shape
        self.target_imsize = tuple(target_imsize)
        # Sory by area to paste in largest faces last
        area = (self.boxes[:, 2] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 1]).view(-1)
        idx = area.argsort(descending=False)
        self.boxes = self.boxes[idx]
        self.fdf128_expand = fdf128_expand
        self.orig_keypoints = keypoints
        if keypoints is not None:
            self.orig_keypoints = self.orig_keypoints[idx]
            assert keypoints.shape == (len(boxes_ltrb), 17, 2) or \
                keypoints.shape == (len(boxes_ltrb), 7, 2), keypoints.shape

    def visualize(self, im):
        if len(self) == 0:
            return im
        orig_device = im.device
        for box in self.boxes:
            simple_expand = False if self.fdf128_expand else True
            e_box = torch.from_numpy(expand_bbox_fdf(box.numpy(), im.shape[-2:], simple_expand))
            im = torchvision.utils.draw_bounding_boxes(im.cpu(), e_box[None], colors=(0, 0, 255), width=2)
        im = torchvision.utils.draw_bounding_boxes(im.cpu(), self.boxes, colors=(255, 0, 0), width=2)
        if self.orig_keypoints is not None:
            _, _, connectivity = vis_utils.get_coco_keypoints()
            connectivity = connectivity[:5]
            im = vis_utils.draw_keypoints(im, self.orig_keypoints, radius=1, connectivity=connectivity, width=3)

        return im.to(device=orig_device)

    def get_crop(self, idx: int, im):
        assert idx < len(self)
        box = self.boxes[idx].numpy()
        simple_expand = False if self.fdf128_expand else True
        expanded_boxes = expand_bbox_fdf(box, im.shape[-2:], simple_expand)
        im = cut_pad_resize(im, expanded_boxes, self.target_imsize, fdf_resize=True)

        # Find the square mask corresponding to box.
        box_mask = box.copy().astype(float)
        box_mask[[0, 2]] -= expanded_boxes[0]
        box_mask[[1, 3]] -= expanded_boxes[1]

        width = expanded_boxes[2] - expanded_boxes[0]
        resize_factor = self.target_imsize[0] / width
        box_mask = (box_mask * resize_factor).astype(int)
        mask = torch.ones((1, *self.target_imsize), device=im.device, dtype=torch.float32)
        crop_box(mask, box_mask).fill_(0)
        if self.orig_keypoints is None:
            return dict(
                img=im[None], mask=mask[None],
                boxes=torch.from_numpy(expanded_boxes).view(1, -1))

        keypoint = self.orig_keypoints[idx, :7, :2].clone()
        keypoint[:, 0] -= expanded_boxes[0]
        keypoint[:, 1] -= expanded_boxes[1]
        w = expanded_boxes[2] - expanded_boxes[0]
        keypoint /= w
        keypoint = keypoint.clamp(0, 1)
        keypoint = torch.cat((keypoint, torch.ones(
            keypoint.shape[0], 1, device=keypoint.device, dtype=keypoint.dtype)), dim=-1)
        return dict(
            img=im[None], mask=mask[None],
            boxes=torch.from_numpy(expanded_boxes).view(1, -1),
            keypoints=keypoint[None])

    def __len__(self):
        return len(self.boxes)

    @staticmethod
    def from_state_dict(state_dict, **kwargs):
        return FaceDetection(
            state_dict["boxes"].cpu(),
            keypoints=state_dict["orig_keypoints"] if "orig_keypoints" in state_dict else None,
            **kwargs)

    def state_dict(self, **kwargs):
        return dict(
            boxes=self.boxes,
            cls=self.__class__,
            orig_keypoints=self.orig_keypoints)

    def pre_process(self):
        pass


def remove_dilate_in_pad(mask: torch.Tensor, exp_box, orig_imshape):
    """
    Dilation happens after padding, which could place dilation in the padded area.
    Remove this.
    """
    x0, y0, x1, y1 = exp_box
    H, W = orig_imshape
    # Padding in original image space
    p_y0 = max(0, -y0)
    p_y1 = max(y1 - H, 0)
    p_x0 = max(0, -x0)
    p_x1 = max(x1 - W, 0)
    resize_ratio = mask.shape[-2] / (y1 - y0)
    p_x0, p_y0, p_x1, p_y1 = [(_ * resize_ratio).floor().long() for _ in [p_x0, p_y0, p_x1, p_y1]]
    mask[..., :p_y0, :] = 0
    mask[..., :p_x0] = 0
    mask[..., mask.shape[-2] - p_y1:, :] = 0
    mask[..., mask.shape[-1] - p_x1:] = 0


class CSEPersonDetection:

    def __init__(self,
                 segmentation, cse_dets,
                 target_imsize,
                 exp_bbox_cfg, exp_bbox_filter,
                 dilation_percentage: float,
                 embed_map: torch.Tensor,
                 orig_imshape_CHW,
                 normalize_embedding: bool) -> None:
        self.segmentation = segmentation
        self.cse_dets = cse_dets
        self.target_imsize = list(target_imsize)
        self.pre_processed = False
        self.exp_bbox_cfg = exp_bbox_cfg
        self.exp_bbox_filter = exp_bbox_filter
        self.dilation_percentage = dilation_percentage
        self.embed_map = embed_map
        self.embed_map_cpu = embed_map.cpu()
        self.normalize_embedding = normalize_embedding
        if self.normalize_embedding:
            embed_map_mean = self.embed_map.mean(dim=0, keepdim=True)
            embed_map_rstd = ((self.embed_map - embed_map_mean).square().mean(dim=0, keepdim=True) + 1e-8).rsqrt()
            self.embed_map_normalized = (self.embed_map - embed_map_mean) * embed_map_rstd
        self.orig_imshape_CHW = orig_imshape_CHW

    @torch.no_grad()
    def pre_process(self):
        if self.pre_processed:
            return
        boxes = initialize_cse_boxes(self.segmentation, self.cse_dets["bbox_XYXY"]).cpu()
        expanded_boxes = []
        included_boxes = []
        for i in range(len(boxes)):
            exp_box = get_expanded_bbox(
                boxes[i], self.orig_imshape_CHW[1:], self.segmentation[i], **self.exp_bbox_cfg,
                target_aspect_ratio=self.target_imsize[0] / self.target_imsize[1])
            if not include_box(exp_box, imshape=self.orig_imshape_CHW[1:], **self.exp_bbox_filter):
                continue
            included_boxes.append(i)
            expanded_boxes.append(exp_box)
        expanded_boxes = torch.LongTensor(expanded_boxes).view(-1, 4)
        self.segmentation = self.segmentation[included_boxes]
        self.cse_dets = {k: v[included_boxes] for k, v in self.cse_dets.items()}

        self.mask = torch.empty((len(expanded_boxes), *self.target_imsize), device=tops.get_device(), dtype=torch.bool)
        area = self.segmentation.sum(dim=[1, 2]).view(len(expanded_boxes))
        for i, box in enumerate(expanded_boxes):
            self.mask[i] = cut_pad_resize(self.segmentation[i:i + 1], box, self.target_imsize)[0]

        dilation_kernel = get_kernel(
            int((self.target_imsize[0] * self.target_imsize[1])**0.5 * self.dilation_percentage))
        self.maskrcnn_mask = self.mask.clone().logical_not()[:, None]
        self.mask = utils.binary_dilation(self.mask[:, None], dilation_kernel)
        for i in range(len(expanded_boxes)):
            remove_dilate_in_pad(self.mask[i], expanded_boxes[i], self.orig_imshape_CHW[1:])
        self.boxes = expanded_boxes.cpu()
        self.dilated_boxes = get_dilated_boxes(self.boxes, self.mask)

        self.pre_processed = True
        self.n_detections = len(self.boxes)
        self.mask = self.mask.logical_not()

        E_mask = torch.zeros((self.n_detections, 1, *self.target_imsize), device=self.mask.device, dtype=torch.bool)
        self.vertices = torch.zeros_like(E_mask, dtype=torch.long)
        for i in range(self.n_detections):
            E_, E_mask[i] = transform_embedding(
                self.cse_dets["instance_embedding"][i],
                self.cse_dets["instance_segmentation"][i],
                self.boxes[i],
                self.cse_dets["bbox_XYXY"][i].cpu(),
                self.target_imsize
            )
            self.vertices[i] = utils.from_E_to_vertex(
                E_[None], E_mask[i:i + 1].logical_not(), self.embed_map).squeeze()[None]
        self.E_mask = E_mask

        sorted_idx = torch.argsort(area, descending=False)
        self.mask = self.mask[sorted_idx]
        self.boxes = self.boxes[sorted_idx.cpu()]
        self.vertices = self.vertices[sorted_idx]
        self.E_mask = self.E_mask[sorted_idx]
        self.maskrcnn_mask = self.maskrcnn_mask[sorted_idx]

    def get_crop(self, idx: int, im):
        self.pre_process()
        assert idx < len(self)
        box = self.boxes[idx]
        mask = self.mask[idx]
        im = cut_pad_resize(im, box, self.target_imsize).unsqueeze(0)

        vertices_ = self.vertices[idx]
        E_mask_ = self.E_mask[idx].float()
        if self.normalize_embedding:
            embedding = self.embed_map_normalized[vertices_.squeeze(dim=0)].permute(2, 0, 1) * E_mask_
        else:
            embedding = self.embed_map[vertices_.squeeze(dim=0)].permute(2, 0, 1) * E_mask_

        return dict(
            img=im,
            mask=mask.float()[None],
            boxes=box.reshape(1, -1),
            E_mask=E_mask_[None],
            vertices=vertices_[None],
            embed_map=self.embed_map,
            embedding=embedding[None],
            maskrcnn_mask=self.maskrcnn_mask[idx].float()[None]
        )

    def __len__(self):
        self.pre_process()
        return self.n_detections

    def state_dict(self, after_preprocess=False):
        """
            The processed annotations occupy more space than the original detections.
        """
        if not after_preprocess:
            return {
                "combined_segmentation": self.segmentation.bool(),
                "cse_instance_segmentation": self.cse_dets["instance_segmentation"].bool(),
                "cse_instance_embedding": self.cse_dets["instance_embedding"],
                "cse_bbox_XYXY": self.cse_dets["bbox_XYXY"].long(),
                "cls": self.__class__,
                "orig_imshape_CHW": self.orig_imshape_CHW
            }
        self.pre_process()
        def compress_bool(x): return torch.from_numpy(np.packbits(x.bool().cpu().numpy()))
        return dict(
            E_mask=compress_bool(self.E_mask),
            mask=compress_bool(self.mask),
            maskrcnn_mask=compress_bool(self.maskrcnn_mask),
            vertices=self.vertices.to(torch.int16).cpu(),
            cls=self.__class__,
            boxes=self.boxes,
            orig_imshape_CHW=self.orig_imshape_CHW,
        )

    @staticmethod
    def from_state_dict(
            state_dict, embed_map,
            post_process_cfg, **kwargs):
        after_preprocess = "segmentation" not in state_dict and "combined_segmentation" not in state_dict
        if after_preprocess:
            detection = CSEPersonDetection(
                segmentation=None, cse_dets=None, embed_map=embed_map,
                orig_imshape_CHW=state_dict["orig_imshape_CHW"],
                **post_process_cfg)
            detection.vertices = tops.to_cuda(state_dict["vertices"].long())
            numel = np.prod(detection.vertices.shape)

            def unpack_bool(x):
                x = torch.from_numpy(np.unpackbits(x.numpy(), count=numel))
                return x.view(*detection.vertices.shape)
            detection.E_mask = tops.to_cuda(unpack_bool(state_dict["E_mask"]))
            detection.mask = tops.to_cuda(unpack_bool(state_dict["mask"]))
            detection.maskrcnn_mask = tops.to_cuda(unpack_bool(state_dict["maskrcnn_mask"]))
            detection.n_detections = len(detection.mask)
            detection.pre_processed = True

            if isinstance(state_dict["boxes"], np.ndarray):
                state_dict["boxes"] = torch.from_numpy(state_dict["boxes"])
            detection.boxes = state_dict["boxes"]
            return detection

        cse_dets = dict(
            instance_segmentation=state_dict["cse_instance_segmentation"],
            instance_embedding=state_dict["cse_instance_embedding"],
            embed_map=embed_map,
            bbox_XYXY=state_dict["cse_bbox_XYXY"])
        cse_dets = {k: tops.to_cuda(v) for k, v in cse_dets.items()}

        segmentation = state_dict["combined_segmentation"]
        return CSEPersonDetection(
            segmentation, cse_dets, embed_map=embed_map,
            orig_imshape_CHW=state_dict["orig_imshape_CHW"],
            **post_process_cfg)

    def visualize(self, im):
        self.pre_process()
        if len(self) == 0:
            return im
        im = vis_utils.draw_cropped_masks(
            im.cpu(), self.mask.cpu(), self.boxes, visualize_instances=False)
        E = self.embed_map_cpu[self.vertices.long().cpu()].squeeze(1).permute(0, 3, 1, 2)
        im = vis_utils.draw_cse_all(
            E, self.E_mask.squeeze(1).bool().cpu(), im,
            self.boxes, self.embed_map_cpu)
        im = torchvision.utils.draw_bounding_boxes(im, self.boxes, colors=(255, 0, 0), width=2)
        return im


def shift_and_preprocess_keypoints(keypoints: torch.Tensor, boxes):
    keypoints = keypoints.clone()
    N = boxes.shape[0]
    tops.assert_shape(keypoints, (N, None, 3))
    tops.assert_shape(boxes, (N, 4))
    x0, y0, x1, y1 = [_.view(-1, 1) for _ in boxes.T]

    w = x1 - x0
    h = y1 - y0
    keypoints[:, :, 0] = (keypoints[:, :, 0] - x0) / w
    keypoints[:, :, 1] = (keypoints[:, :, 1] - y0) / h
    def check_outside(x): return (x < 0).logical_or(x > 1)
    is_outside = check_outside(keypoints[:, :, 0]).logical_or(check_outside(keypoints[:, :, 1]))
    keypoints[:, :, 2] = keypoints[:, :, 2] > 0
    keypoints[:, :, 2] = (keypoints[:, :, 2] > 0).logical_and(is_outside.logical_not())
    return keypoints


class PersonDetection:

    def __init__(
            self,
            segmentation,
            target_imsize,
            exp_bbox_cfg, exp_bbox_filter,
            dilation_percentage: float,
            orig_imshape_CHW,
            kp_vis_thr=None,
            keypoints=None,
            insert_joint_map=False,
            **kwargs) -> None:
        self.segmentation = segmentation
        self.target_imsize = list(target_imsize)
        self.pre_processed = False
        self.exp_bbox_cfg = exp_bbox_cfg
        self.exp_bbox_filter = exp_bbox_filter
        self.dilation_percentage = dilation_percentage
        self.orig_imshape_CHW = orig_imshape_CHW
        self.orig_keypoints = keypoints
        if keypoints is not None:
            assert kp_vis_thr is not None
            self.kp_vis_thr = kp_vis_thr
        if insert_joint_map:
            self.joint_map_transform = InsertJointMap(target_imsize)
        

    @torch.no_grad()
    def pre_process(self):
        if self.pre_processed:
            return
        boxes = masks_to_boxes(self.segmentation).cpu()
        expanded_boxes = []
        included_boxes = []
        for i in range(len(boxes)):
            exp_box = get_expanded_bbox(
                boxes[i], self.orig_imshape_CHW[1:], self.segmentation[i], **self.exp_bbox_cfg,
                target_aspect_ratio=self.target_imsize[0] / self.target_imsize[1])
            if not include_box(exp_box, imshape=self.orig_imshape_CHW[1:], **self.exp_bbox_filter):
                continue
            included_boxes.append(i)
            expanded_boxes.append(exp_box)
        expanded_boxes = torch.LongTensor(expanded_boxes).view(-1, 4)
        self.segmentation = self.segmentation[included_boxes]
        if self.orig_keypoints is not None:
            self.keypoints = self.orig_keypoints[included_boxes].clone()
            self.keypoints[:, :, 2] = self.keypoints[:, :, 2] >= self.kp_vis_thr
        area = self.segmentation.sum(dim=[1, 2]).view(len(expanded_boxes)).cpu()
        self.mask = torch.empty((len(expanded_boxes), *self.target_imsize), device=tops.get_device(), dtype=torch.bool)
        for i, box in enumerate(expanded_boxes):
            self.mask[i] = cut_pad_resize(self.segmentation[i:i + 1], box, self.target_imsize)[0]
        if self.orig_keypoints is not None:
            self.keypoints = shift_and_preprocess_keypoints(self.keypoints, expanded_boxes)
        dilation_kernel = get_kernel(
            int((self.target_imsize[0] * self.target_imsize[1])**0.5 * self.dilation_percentage))
        self.maskrcnn_mask = self.mask.clone().logical_not()[:, None]
        self.mask = utils.binary_dilation(self.mask[:, None], dilation_kernel)
        for i in range(len(expanded_boxes)):
            remove_dilate_in_pad(self.mask[i], expanded_boxes[i], self.orig_imshape_CHW[1:])
        self.boxes = expanded_boxes
        self.dilated_boxes = get_dilated_boxes(self.boxes, self.mask)

        self.pre_processed = True
        self.n_detections = len(self.boxes)
        self.mask = self.mask.logical_not()

        sorted_idx = torch.argsort(area, descending=False)
        self.mask = self.mask[sorted_idx]
        self.boxes = self.boxes[sorted_idx.cpu()]
        self.segmentation = self.segmentation[sorted_idx]
        self.maskrcnn_mask = self.maskrcnn_mask[sorted_idx]
        if self.orig_keypoints is not None:
            self.keypoints = self.keypoints[sorted_idx.cpu()]

    def get_crop(self, idx: int, im: torch.Tensor):
        assert idx < len(self)
        self.pre_process()
        box = self.boxes[idx]
        mask = self.mask[idx][None].float()
        im = cut_pad_resize(im, box, self.target_imsize).unsqueeze(0)
        batch = dict(
            img=im, mask=mask, boxes=box.reshape(1, -1),
            maskrcnn_mask=self.maskrcnn_mask[idx][None].float())
        if self.orig_keypoints is not None:
            batch["keypoints"] = self.keypoints[idx:idx + 1]
        if hasattr(self, "joint_map_transform"):
            batch = self.joint_map_transform(batch)
        return batch

    def __len__(self):
        self.pre_process()
        return self.n_detections

    def state_dict(self, **kwargs):
        return dict(
            segmentation=self.segmentation.bool(),
            cls=self.__class__,
            orig_imshape_CHW=self.orig_imshape_CHW,
            keypoints=self.orig_keypoints
        )

    @staticmethod
    def from_state_dict(
            state_dict,
            post_process_cfg, **kwargs):
        return PersonDetection(
            state_dict["segmentation"],
            orig_imshape_CHW=state_dict["orig_imshape_CHW"],
            **post_process_cfg,
            keypoints=state_dict["keypoints"])

    def visualize(self, im):
        self.pre_process()
        im = im.cpu()
        if len(self) == 0:
            return im
        im = vis_utils.draw_cropped_masks(im.clone(), self.mask.cpu(), self.boxes, visualize_instances=False)
        if self.orig_keypoints is not None:
            im = vis_utils.draw_cropped_keypoints(im, self.keypoints, self.boxes)
        im = torchvision.utils.draw_bounding_boxes(im, self.boxes, colors=(255, 0, 0), width=2)
        return im


def get_dilated_boxes(exp_bbox: torch.LongTensor, mask):
    """
        mask: resized mask
    """
    assert exp_bbox.shape[0] == mask.shape[0]
    boxes = masks_to_boxes(mask.squeeze(1)).cpu()
    H, W = exp_bbox[:, 3] - exp_bbox[:, 1], exp_bbox[:, 2] - exp_bbox[:, 0]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * W[:, None] / mask.shape[-1]).long()
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * H[:, None] / mask.shape[-2]).long()
    boxes[:, [0, 2]] += exp_bbox[:, 0:1]
    boxes[:, [1, 3]] += exp_bbox[:, 1:2]
    return boxes
