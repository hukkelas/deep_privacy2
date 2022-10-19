from ..FB_cse_mask_face import anonymizer, detector, common

detector.score_threshold = .1
detector.face_detector_cfg.confidence_threshold = .5
detector.cse_cfg.score_thres = 0.3
anonymizer.generators.face_G_cfg = None
anonymizer.generators.person_G_cfg = "configs/generators/dummy/pixelation16.py"
anonymizer.generators.cse_person_G_cfg = "configs/generators/dummy/pixelation16.py"