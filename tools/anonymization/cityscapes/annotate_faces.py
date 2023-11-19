import numpy as np
import cv2
import tqdm
import json
from pathlib import Path
from face_detection import build_detector
from face_detection.dsfd import DSFDDetector
from detectron2.data.detection_utils import _apply_exif_orientation
from PIL import Image
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from detectron2.structures.masks import polygons_to_bitmask
from ..coco.annotate_faces import get_matches

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


def draw_keypoints(im, keypoints):
    for kp in keypoints:
        for x0, y0, visible in kp:
            if visible == 0:
                continue
            cv2.circle(im, (x0, y0), 1, color=(0, 255,0), thickness=2)

ddir = Path("/mnt/work2/haakohu/datasets/cityscapes/")
def iterate_dataset():
    dataset = load_cityscapes_instances(ddir.joinpath("leftImg8bit/train"), ddir.joinpath("gtFine/train"))

    for sample in tqdm.tqdm(dataset): # dict_keys(['file_name', 'image_id', 'height', 'width', 'annotations'])
        masks = []
        for i, instance_annotation in enumerate(sample["annotations"]): # dict_keys(['iscrowd', 'category_id', 'segmentation', 'bbox', 'bbox_mode'])
            # Detectron2 filters out all instances part of crowd
            # In keypoint annotation I did not filter out crowd label
            assert isinstance(instance_annotation["iscrowd"], bool)
            if instance_annotation["iscrowd"]:
                continue
            if int(instance_annotation["category_id"]) != 0:
                continue
            mask = polygons_to_bitmask(instance_annotation["segmentation"], height=sample["height"], width=sample["width"])
            masks.append(mask)
        if len(masks) == 0:
            yield sample["file_name"], sample["image_id"], None, None, None
            continue
        im = Image.open(sample["file_name"])
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        im = np.array(im)
        masks = np.stack(masks)
        yield sample["file_name"], sample["image_id"], im, masks, orig_im_mode

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def annotate_cityscapes(detector: DSFDDetector):
    target_path = ddir.joinpath("face_boxes_train.json")
    #print(dataset[0]["annotations"][0].keys(())
    all_results = dict()
    n_instances = 0
    n_boxes = 0
    for file_name, image_id, im, masks, orig_im_mode in iterate_dataset():
        if im is None:
            continue
        boxes = detector.detect(im)
        n_instances += len(masks)
        if len(boxes) == 0:
            continue
        matches, _ = get_matches(boxes, masks)
        n_boxes += len(matches)
        boxes = [boxes[i].tolist() for i, j in matches]
        all_results[image_id] = boxes
    print(f"There are {n_instances}, where {n_boxes} are detected with face detector.")
    with open(target_path, "w") as fp:
        json.dump(all_results, fp)


if __name__ == "__main__":
    detector = build_detector(
        clip_boxes=True,
        confidence_threshold=0.2
        )
    annotate_cityscapes(
        detector,
    )