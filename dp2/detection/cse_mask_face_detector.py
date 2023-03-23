import torch
import lzma
import tops
from pathlib import Path
from dp2.detection.base import BaseDetector
from .utils import combine_cse_maskrcnn_dets
from face_detection import build_detector as build_face_detector
from .models.cse import CSEDetector
from .models.mask_rcnn import MaskRCNNDetector
from .structures import CSEPersonDetection, VehicleDetection, FaceDetection, PersonDetection
from tops import logger


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


class CSeMaskFaceDetector(BaseDetector):

    def __init__(
            self,
            mask_rcnn_cfg,
            face_detector_cfg: dict,
            cse_cfg: dict,
            face_post_process_cfg: dict,
            cse_post_process_cfg,
            score_threshold: float,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mask_rcnn = MaskRCNNDetector(**mask_rcnn_cfg, score_thres=score_threshold)
        if "confidence_threshold" not in face_detector_cfg:
            face_detector_cfg["confidence_threshold"] = score_threshold
        if "score_thres" not in cse_cfg:
            cse_cfg["score_thres"] = score_threshold
        self.cse_detector = CSEDetector(**cse_cfg)
        self.face_detector = build_face_detector(**face_detector_cfg, clip_boxes=True)
        self.cse_post_process_cfg = cse_post_process_cfg
        self.face_mean = tops.to_cuda(torch.from_numpy(self.face_detector.mean).view(3, 1, 1))
        self.mask_cse_iou_combine_threshold = self.cse_post_process_cfg.pop("iou_combine_threshold")
        self.face_post_process_cfg = face_post_process_cfg

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _detect_faces(self, im: torch.Tensor):
        H, W = im.shape[1:]
        im = im.float() - self.face_mean
        im = self.face_detector.resize(im[None], 1.0)
        boxes_XYXY = self.face_detector._batched_detect(im)[0][:, :-1]  # Remove score
        boxes_XYXY[:, [0, 2]] *= W
        boxes_XYXY[:, [1, 3]] *= H
        return boxes_XYXY.round().long()

    def load_from_cache(self, cache_path: Path):
        logger.log(f"Loading detection from cache path: {cache_path}",)
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        kwargs = dict(
            post_process_cfg=self.cse_post_process_cfg,
            embed_map=self.cse_detector.embed_map,
            **self.face_post_process_cfg
        )
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        maskrcnn_dets = self.mask_rcnn(im)
        cse_dets = self.cse_detector(im)
        embed_map = self.cse_detector.embed_map
        print("Calling face detector.")
        face_boxes = self._detect_faces(im).cpu()
        maskrcnn_person = {
            k: v[maskrcnn_dets["is_person"]] for k, v in maskrcnn_dets.items()
        }
        maskrcnn_other = {
            k: v[maskrcnn_dets["is_person"].logical_not()] for k, v in maskrcnn_dets.items()
        }
        maskrcnn_other = VehicleDetection(maskrcnn_other["segmentation"])
        combined_segmentation, cse_dets, matches = combine_cse_maskrcnn_dets(
            maskrcnn_person["segmentation"], cse_dets, self.mask_cse_iou_combine_threshold)

        persons_with_cse = CSEPersonDetection(
            combined_segmentation, cse_dets, **self.cse_post_process_cfg,
            embed_map=embed_map, orig_imshape_CHW=im.shape
        )
        persons_with_cse.pre_process()
        not_matched = [i for i in range(maskrcnn_person["segmentation"].shape[0]) if i not in matches[:, 0]]
        persons_without_cse = PersonDetection(
            maskrcnn_person["segmentation"][not_matched], **self.cse_post_process_cfg,
            orig_imshape_CHW=im.shape
        )
        persons_without_cse.pre_process()

        face_boxes_covered = box1_inside_box2(face_boxes, persons_with_cse.dilated_boxes).logical_or(
            box1_inside_box2(face_boxes, persons_without_cse.dilated_boxes)
        )
        face_boxes = face_boxes[face_boxes_covered.logical_not()]
        face_boxes = FaceDetection(face_boxes, **self.face_post_process_cfg)

        # Order matters. The anonymizer will anonymize FIFO.
        # Later detections will overwrite.
        all_detections = [face_boxes, maskrcnn_other, persons_without_cse, persons_with_cse]
        return all_detections
