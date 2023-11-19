import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads import CascadeROIHeads, StandardROIHeads
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.structures import Instances
from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.config import LazyCall as L
from PIL import Image
import tops
import functools
from torchvision.transforms.functional import resize


def get_rn50_fpn_keypoint_rcnn(weight_path: str):
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
    from detectron2.layers import ShapeSpec
    model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
    model.roi_heads.update(
        num_classes=1,
        keypoint_in_features=["p2", "p3", "p4", "p5"],
        keypoint_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_keypoints=17,
            conv_dims=[512] * 8,
            loss_normalizer="visible",
        ),
    )

    # Detectron1 uses 2000 proposals per-batch, but this option is per-image in detectron2.
    # 1000 proposals per-image is found to hurt box AP.
    # Therefore we increase it to 1500 per-image.
    model.proposal_generator.post_nms_topk = (1500, 1000)

    # Keypoint AP degrades (though box AP improves) when using plain L1 loss
    model.roi_heads.box_predictor.smooth_l1_beta = 0.5
    model = instantiate(model)

    dataloader = model_zoo.get_config("common/data/coco_keypoint.py").dataloader
    test_transform = instantiate(dataloader.test.mapper.augmentations)
    DetectionCheckpointer(model).load(weight_path)
    return model, test_transform


models = {
    "rn50_fpn_maskrcnn": functools.partial(get_rn50_fpn_keypoint_rcnn, weight_path="https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/532a57f3-594b-4ec9-a6db-ef2e328ad60ae337668e-a83c-4222-9fa0-cec6f91adf4841b9a42e-a28e-403e-8b96-d55ac443b8c6")
}


class KeypointMaskRCNN:

    def __init__(self, model_name: str, score_threshold: float) -> None:
        assert model_name in models, f"Did not find {model_name} in models"
        model, test_transform = models[model_name]()
        self.model = model.eval().to(tops.get_device())
        if isinstance(self.model.roi_heads, CascadeROIHeads):
            for head in self.model.roi_heads.box_predictors:
                assert hasattr(head, "test_score_thresh")
                head.test_score_thresh = score_threshold
        else:
            assert isinstance(self.model.roi_heads, StandardROIHeads)
            assert hasattr(self.model.roi_heads.box_predictor, "test_score_thresh")
            self.model.roi_heads.box_predictor.test_score_thresh = score_threshold

        self.test_transform = test_transform
        assert len(self.test_transform) == 1
        self.test_transform = self.test_transform[0]
        assert isinstance(self.test_transform, ResizeShortestEdge)
        assert self.test_transform.interp == Image.BILINEAR
        self.image_format = self.model.input_format

    def resize_im(self, im):
        H, W = im.shape[-2:]
        if self.test_transform.is_range:
            size = np.random.randint(
                self.test_transform.short_edge_length[0], self.test_transform.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.test_transform.short_edge_length)
        newH, newW = ResizeShortestEdge.get_output_shape(H, W, size, self.test_transform.max_size)
        return resize(
            im, (newH, newW), antialias=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(self, im: torch.Tensor):
        assert im.ndim == 3
        if self.image_format == "BGR":
            im = im.flip(0)
        H, W = im.shape[-2:]
        im = im.float()
        im = self.resize_im(im)

        inputs = dict(image=im, height=H, width=W)
        # instances contains
        # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks', 'pred_keypoints', 'pred_keypoint_heatmaps'])
        instances = self.model([inputs])[0]["instances"]
        return dict(
            scores=instances.get("scores").cpu(),
            segmentation=instances.get("pred_masks").cpu(),
            keypoints=instances.get("pred_keypoints").cpu()
        )
