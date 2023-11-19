from dp2.anonymizer import Anonymizer
from dp2.detection.models.vit_pose_maskrcnn import MaskRCNNVitPose
from ..defaults import common
from tops.config import LazyCall as L


detector = L(MaskRCNNVitPose)(
    mask_rcnn_cfg=dict(),
    post_process_cfg=dict(
        target_imsize=(288, 160),
        exp_bbox_cfg=dict(percentage_background=0.3, axis_minimum_expansion=0.1),
        exp_bbox_filter=dict(
            minimum_area=32 * 32, min_bbox_ratio_inside=0, aspect_ratio_range=[0, 99999]
        ),
        dilation_percentage=0.02,
        kp_vis_thr=0.05,
        insert_joint_map=True,
    ),
    score_threshold=0.3,
    cache_directory=common.output_dir.joinpath("maskrcnn_vitpose_cache"),
)

anonymizer = L(Anonymizer)(
    detector="${detector}",
    person_G_cfg="configs/fdh/triagan/L_im288.py",
)
