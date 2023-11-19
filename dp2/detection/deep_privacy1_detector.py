import torch
import tops
import lzma
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from .base import BaseDetector
from face_detection import build_detector as build_face_detector
from .structures import FaceDetection
from tops import logger
from pathlib import Path

def is_keypoint_within_bbox(x0, y0, x1, y1, keypoint):
    keypoint = keypoint[:3, :]  # only nose + eyes are relevant
    kp_X = keypoint[:, 0]
    kp_Y = keypoint[:, 1]
    within_X = (kp_X >= x0).all() and (kp_X <= x1).all()
    within_Y = (kp_Y >= y0).all() and (kp_Y <= y1).all()
    return within_X and within_Y


def match_bbox_keypoint(bounding_boxes, keypoints):
    """
        bounding_boxes shape: [N, 5]
        keypoints: [N persons, K keypoints, (x, y)]
    """
    if len(bounding_boxes) == 0 or len(keypoints) == 0:
        return torch.empty((0, 4)), torch.empty((0, 7, 2))
    assert bounding_boxes.shape[1] == 4,\
        f"Shape was : {bounding_boxes.shape}"
    assert keypoints.shape[-1] == 2,\
        f"Expected (x,y) in last axis, got: {keypoints.shape}"
    assert keypoints.shape[1] in (5, 7),\
        f"Expeted 5 or 7 keypoints. Keypoint shape was: {keypoints.shape}"

    matches = []
    for bbox_idx, bbox in enumerate(bounding_boxes):
        keypoint = None
        for kp_idx, keypoint in enumerate(keypoints):
            if kp_idx in (x[1] for x in matches):
                continue
            if is_keypoint_within_bbox(*bbox, keypoint):
                matches.append((bbox_idx, kp_idx))
                break
    keypoint_idx = [x[1] for x in matches]
    bbox_idx = [x[0] for x in matches]
    return bounding_boxes[bbox_idx], keypoints[keypoint_idx]


class DeepPrivacy1Detector(BaseDetector):

    def __init__(self,
                 keypoint_threshold: float,
                 face_detector_cfg,
                 score_threshold: float,
                 face_post_process_cfg,
                 **kwargs):
        super().__init__(**kwargs)
        self.keypoint_detector = tops.to_cuda(keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).eval())
        self.keypoint_threshold = keypoint_threshold
        self.face_detector = build_face_detector(**face_detector_cfg, confidence_threshold=score_threshold)
        self.face_mean = tops.to_cuda(torch.from_numpy(self.face_detector.mean).view(3, 1, 1))
        self.face_post_process_cfg = face_post_process_cfg

    @torch.no_grad()
    def _detect_faces(self, im: torch.Tensor):
        H, W = im.shape[1:]
        im = im.float() - self.face_mean
        im = self.face_detector.resize(im[None], 1.0)
        boxes_XYXY = self.face_detector._batched_detect(im)[0][:, :-1]  # Remove score
        boxes_XYXY[:, [0, 2]] *= W
        boxes_XYXY[:, [1, 3]] *= H
        return boxes_XYXY.round().long().cpu()

    @torch.no_grad()
    def _detect_keypoints(self, img: torch.Tensor):
        img = img.float() / 255
        outputs = self.keypoint_detector([img])

        # Shape: [N persons, K keypoints, (x,y,visibility)]
        keypoints = outputs[0]["keypoints"]
        scores = outputs[0]["scores"]
        assert list(scores) == sorted(list(scores))[::-1]
        mask = scores >= self.keypoint_threshold
        keypoints = keypoints[mask, :, :2]
        return keypoints[:, :7, :2]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        face_boxes = self._detect_faces(im)
        keypoints = self._detect_keypoints(im)
        face_boxes, keypoints = match_bbox_keypoint(face_boxes, keypoints)
        face_boxes = FaceDetection(face_boxes, **self.face_post_process_cfg, keypoints=keypoints)
        return [face_boxes]

    def load_from_cache(self, cache_path: Path):
        logger.log(f"Loading detection from cache path: {cache_path}",)
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        kwargs = self.face_post_process_cfg
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]