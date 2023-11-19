import torch
import lzma
from pathlib import Path
from dp2.detection.base import BaseDetector
from .mask_rcnn import MaskRCNNDetector
from ..structures import PersonDetection
from tops import logger
from .vit_pose.vit_pose import VitPoseModel
from ..utils import masks_to_boxes


def box1_inside_box2(box1: torch.Tensor, box2: torch.Tensor):
    assert len(box1.shape) == 2
    assert len(box2.shape) == 2
    box1_inside = torch.zeros(box1.shape[0], device=box1.device, dtype=torch.bool)
    # This can be batched
    for i, box in enumerate(box1):
        is_outside_lefttop = (box[None, [0, 1]] <= box2[:, [0, 1]]).any(dim=1)
        is_outside_rightbot = (box[None, [2, 3]] >= box2[:, [2, 3]]).any(dim=1)
        is_outside = is_outside_lefttop.logical_or(is_outside_rightbot)
        box1_inside[i] = is_outside.logical_not().any()
    return box1_inside


class MaskRCNNVitPose(BaseDetector):

    def __init__(
            self,
            mask_rcnn_cfg,
            post_process_cfg,
            score_threshold: float,
            **kwargs
    ) -> None:
        super().__init__(kwargs["cache_directory"])
        self.mask_rcnn = MaskRCNNDetector(**mask_rcnn_cfg, score_thres=score_threshold)
        self.vit_pose = VitPoseModel("vit_huge")

        self.post_process_cfg = post_process_cfg

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_from_cache(self, cache_path: Path):
        logger.log(f"Loading detection from cache path: {cache_path}",)
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        kwargs = dict(
            post_process_cfg=self.post_process_cfg,
        )
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        maskrcnn_dets = self.mask_rcnn(im)

        maskrcnn_person = {
            k: v[maskrcnn_dets["is_person"]] for k, v in maskrcnn_dets.items()
        }
        boxes = masks_to_boxes(maskrcnn_person["segmentation"])
        keypoints = self.vit_pose(im, boxes).cpu()
        keypoints[:, :, -1] = keypoints[:, :, -1] >= 0.3
        persons_without_cse = PersonDetection(
            maskrcnn_person["segmentation"], **self.post_process_cfg,
            orig_imshape_CHW=im.shape,
            keypoints=keypoints,
        )
        persons_without_cse.pre_process()

        all_detections = [persons_without_cse]
        return all_detections
