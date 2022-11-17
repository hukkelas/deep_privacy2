from dp2.anonymizer import Anonymizer
from dp2.detection.face_detector import FaceDetector
from ..defaults import common
from tops.config import LazyCall as L


detector = L(FaceDetector)(
    face_detector_cfg=dict(name="DSFDDetector", clip_boxes=True),
    face_post_process_cfg=dict(target_imsize=(128, 128), fdf128_expand=True),
    score_threshold=0.3,
    cache_directory=common.output_dir.joinpath("face_detection_cache")
)


anonymizer = L(Anonymizer)(
    detector="${detector}",
    face_G_cfg="configs/fdf/stylegan_fdf128.py",
)
