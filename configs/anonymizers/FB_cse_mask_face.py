from dp2.anonymizer import Anonymizer
from dp2.detection.cse_mask_face_detector import CSeMaskFaceDetector
from ..defaults import common
from tops.config import LazyCall as L

detector = L(CSeMaskFaceDetector)(
    mask_rcnn_cfg=dict(),
    face_detector_cfg=dict(),
    face_post_process_cfg=dict(target_imsize=(256, 256), fdf128_expand=False),
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
    cache_directory=common.output_dir.joinpath("cse_mask_face_detection_cache")
)

anonymizer = L(Anonymizer)(
    detector="${detector}",
    face_G_cfg="configs/fdf/stylegan.py",
    person_G_cfg="configs/fdh/styleganL_nocse.py",
    cse_person_G_cfg="configs/fdh/styleganL.py",
    car_G_cfg="configs/generators/dummy/pixelation8.py"
)
