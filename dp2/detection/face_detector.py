import torch
import lzma
import tops
from pathlib import Path
from dp2.detection.base import BaseDetector
from face_detection import build_detector as build_face_detector
from .structures import FaceDetection
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


class FaceDetector(BaseDetector):

    def __init__(
            self,
            face_detector_cfg: dict,
            score_threshold: float,
            face_post_process_cfg: dict,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.face_detector = build_face_detector(**face_detector_cfg, confidence_threshold=score_threshold)
        self.face_mean = tops.to_cuda(torch.from_numpy(self.face_detector.mean).view(3, 1, 1))
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
        return boxes_XYXY.round().long().cpu()

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        face_boxes = self._detect_faces(im)
        face_boxes = FaceDetection(face_boxes, **self.face_post_process_cfg)
        return [face_boxes]

    def load_from_cache(self, cache_path: Path):
        logger.log(f"Loading detection from cache path: {cache_path}")
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp)
        return [
            state["cls"].from_state_dict(state_dict=state, **self.face_post_process_cfg) for state in state_dict
        ]
