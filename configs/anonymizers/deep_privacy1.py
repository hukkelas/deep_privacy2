from .face_fdf128 import anonymizer, common, detector
from dp2.detection.deep_privacy1_detector import DeepPrivacy1Detector
from tops.config import LazyCall as L

anonymizer.update(
    face_G_cfg="configs/fdf/deep_privacy1.py",
)

anonymizer.detector = L(DeepPrivacy1Detector)(
    face_detector_cfg=dict(name="DSFDDetector", clip_boxes=True),
    face_post_process_cfg=dict(target_imsize=(128, 128), fdf128_expand=True),
    score_threshold=0.3,
    keypoint_threshold=0.3,
    cache_directory=common.output_dir.joinpath("deep_privacy1_cache")
)
