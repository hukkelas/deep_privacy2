import torch
import tops
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from typing import Dict
from detectron2.data.transforms import ResizeShortestEdge
from torchvision.transforms.functional import resize


model_urls = {
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",

}


class MaskRCNNDetector:

    def __init__(
            self,
            cfg_name: str = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            score_thres: float = 0.9,
            class_filter=["person"],  # ["car", "bicycle","truck", "bus",  "backpack"]
            fp16_inference: bool = False
    ) -> None:
        cfg = model_zoo.get_config(cfg_name)
        cfg.MODEL.DEVICE = str(tops.get_device())
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
        cfg.freeze()
        self.cfg = cfg
        with tops.logger.capture_log_stdout():
            self.model = build_model(cfg)
            DetectionCheckpointer(self.model).load(model_urls[cfg_name])
        self.model.eval()
        self.input_format = cfg.INPUT.FORMAT
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        self.class_to_keep = set([self.class_names.index(cls_) for cls_ in class_filter])
        self.person_class = self.class_names.index("person")
        self.fp16_inference = fp16_inference
        tops.logger.log("Mask R-CNN built.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def resize_im(self, im):
        H, W = im.shape[1:]
        newH, newW = ResizeShortestEdge.get_output_shape(
            H, W, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        return resize(
            im, (newH, newW), antialias=True)

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        if self.input_format == "BGR":
            im = im.flip(0)
        else:
            assert self.input_format == "RGB"
        H, W = im.shape[-2:]
        im = self.resize_im(im)
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            output = self.model([{"image": im, "height": H, "width": W}])[0]["instances"]
        scores = output.get("scores")
        N = len(scores)
        classes = output.get("pred_classes")
        idx2keep = [i for i in range(N) if classes[i].tolist() in self.class_to_keep]
        classes = classes[idx2keep]
        assert isinstance(output.get("pred_boxes"), Boxes)
        segmentation = output.get("pred_masks")[idx2keep]
        assert segmentation.dtype == torch.bool
        is_person = classes == self.person_class
        return {
            "scores": output.get("scores")[idx2keep],
            "segmentation": segmentation,
            "classes": output.get("pred_classes")[idx2keep],
            "is_person": is_person
        }
