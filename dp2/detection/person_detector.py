import torch
import lzma
from dp2.detection.base import BaseDetector
from .utils import combine_cse_maskrcnn_dets
from .models.cse import CSEDetector
from .models.mask_rcnn import MaskRCNNDetector
from .models.keypoint_maskrcnn import KeypointMaskRCNN
from .structures import CSEPersonDetection, PersonDetection
from pathlib import Path


class CSEPersonDetector(BaseDetector):
    def __init__(
        self,
        score_threshold: float,
        mask_rcnn_cfg: dict,
        cse_cfg: dict,
        cse_post_process_cfg: dict,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mask_rcnn = MaskRCNNDetector(**mask_rcnn_cfg, score_thres=score_threshold)
        self.cse_detector = CSEDetector(**cse_cfg, score_thres=score_threshold)
        self.post_process_cfg = cse_post_process_cfg
        self.iou_combine_threshold = self.post_process_cfg.pop("iou_combine_threshold")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_from_cache(self, cache_path: Path):
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp)
        kwargs = dict(
            post_process_cfg=self.post_process_cfg,
            embed_map=self.cse_detector.embed_map,
        )
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]

    @torch.no_grad()
    def forward(self, im: torch.Tensor, cse_dets=None):
        mask_dets = self.mask_rcnn(im)
        if cse_dets is None:
            cse_dets = self.cse_detector(im)
        segmentation = mask_dets["segmentation"]
        segmentation, cse_dets, _ = combine_cse_maskrcnn_dets(
            segmentation, cse_dets, self.iou_combine_threshold
        )
        det = CSEPersonDetection(
            segmentation=segmentation,
            cse_dets=cse_dets,
            embed_map=self.cse_detector.embed_map,
            orig_imshape_CHW=im.shape,
            **self.post_process_cfg
        )
        return [det]


class MaskRCNNPersonDetector(BaseDetector):
    def __init__(
        self,
        score_threshold: float,
        mask_rcnn_cfg: dict,
        cse_post_process_cfg: dict,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mask_rcnn = MaskRCNNDetector(**mask_rcnn_cfg, score_thres=score_threshold)
        self.post_process_cfg = cse_post_process_cfg

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_from_cache(self, cache_path: Path):
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp)
        kwargs = dict(
            post_process_cfg=self.post_process_cfg,
        )
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        mask_dets = self.mask_rcnn(im)
        segmentation = mask_dets["segmentation"]
        det = PersonDetection(
            segmentation, **self.post_process_cfg, orig_imshape_CHW=im.shape
        )
        return [det]


class KeypointMaskRCNNPersonDetector(BaseDetector):
    def __init__(
        self,
        score_threshold: float,
        mask_rcnn_cfg: dict,
        cse_post_process_cfg: dict,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mask_rcnn = KeypointMaskRCNN(
            **mask_rcnn_cfg, score_threshold=score_threshold
        )
        self.post_process_cfg = cse_post_process_cfg

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_from_cache(self, cache_path: Path):
        with lzma.open(cache_path, "rb") as fp:
            state_dict = torch.load(fp)
        kwargs = dict(
            post_process_cfg=self.post_process_cfg,
        )
        return [
            state["cls"].from_state_dict(**kwargs, state_dict=state)
            for state in state_dict
        ]

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        mask_dets = self.mask_rcnn(im)
        segmentation = mask_dets["segmentation"]
        det = PersonDetection(
            segmentation,
            **self.post_process_cfg,
            orig_imshape_CHW=im.shape,
            keypoints=mask_dets["keypoints"]
        )
        return [det]
