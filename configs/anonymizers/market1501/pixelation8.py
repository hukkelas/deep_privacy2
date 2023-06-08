from ..FB_cse_mask_face import anonymizer, detector, common

detector.score_threshold = .1
detector.face_detector_cfg.confidence_threshold = .5
detector.cse_cfg.score_thres = 0.3
anonymizer.face_G_cfg = None
anonymizer.person_G_cfg = "configs/generators/dummy/pixelation8.py"
anonymizer.cse_person_G_cfg = "configs/generators/dummy/pixelation8.py"