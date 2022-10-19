from dp2.anonymizer import Anonymizer
from dp2.detection.person_detector import CSEPersonDetector
from ..defaults import common
from tops.config import LazyCall as L
from dp2.generator.dummy_generators import MaskOutGenerator


maskout_G = L(MaskOutGenerator)(noise="constant")

detector = L(CSEPersonDetector)(
    mask_rcnn_cfg=dict(),
    cse_cfg=dict(),
    cse_post_process_cfg=dict(
        target_imsize=(288, 160),
        exp_bbox_cfg=dict(percentage_background=0.3, axis_minimum_expansion=.1),
        exp_bbox_filter=dict(minimum_area=32*32, min_bbox_ratio_inside=0, aspect_ratio_range=[0, 99999]),
        iou_combine_threshold=0.4,
        dilation_percentage=0.02,
        normalize_embedding=False
    ),
    score_threshold=0.3,
    cache_directory=common.output_dir.joinpath("cse_person_detection_cache")
)

anonymizer = L(Anonymizer)(
    detector="${detector}",
    cse_person_G_cfg="configs/fdh/styleganL.py",    
)
