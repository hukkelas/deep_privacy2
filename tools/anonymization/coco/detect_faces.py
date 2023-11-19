import numpy as np
import tqdm
import json
from pathlib import Path
from face_detection import build_detector
from face_detection.dsfd import DSFDDetector
import tops
from detectron2.data.detection_utils import _apply_exif_orientation
from PIL import Image
from pycocotools.coco import COCO

def annotate_coco(coco_path: Path, detector: DSFDDetector, target_path: Path):
    face_detection = dict()
    data = COCO(coco_path.joinpath("annotations", "person_keypoints_train2017.json"))
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        if len(annotations) == 0:
            continue
        image_path = coco_path.joinpath("train2017", image_info["file_name"])
        im = Image.open(image_path)
    
        im = _apply_exif_orientation(im)
        im = im.convert("RGB")
        im = np.array(im)
        boxes = detector.detect(im).tolist()
        face_detection[image_id] = boxes
    with open(target_path, "w") as fp:
        json.dump(face_detection, fp)

if __name__ == "__main__":
    detector = build_detector(
        clip_boxes=True,
        confidence_threshold=0.3
        )
    coco_path = Path("/mnt/work2/haakohu/datasets/coco/")
    annotate_coco(
        coco_path,
        detector,
        coco_path.joinpath("initial_detected_boxes_train2017.json")
    )
