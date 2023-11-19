import numpy as np
import cv2
import json
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from .annotate_faces import iterate_dataset, ddir
def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == "__main__":
    target_path = ddir.joinpath("face_boxes_train.json")
    print(target_path)
    dataset = load_cityscapes_instances(ddir.joinpath("leftImg8bit/train"), ddir.joinpath("gtFine/train"))

    with open(target_path, "r") as fp:
        all_results = json.load(fp)
    for file_name, image_id, im, masks, orig_im_mode in iterate_dataset():
        if im is None or image_id not in all_results:
            continue
        boxes = all_results[image_id]
        draw_faces(im, boxes)
        cv2.imshow("test", im)
        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()