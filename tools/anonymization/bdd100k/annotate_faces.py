import numpy as np
import cv2
import tqdm
import json
from pathlib import Path
from face_detection import build_detector
from pycocotools.coco import COCO
from ..coco.annotate_faces import draw_faces, get_matches
from PIL import Image
from detectron2.data.detection_utils import _apply_exif_orientation

def annotate_bdd100k():
    detector = build_detector(
        clip_boxes=True,
        confidence_threshold=0.3
        )
    source_directory = Path("/mnt/work2/haakohu/datasets/bdd100k")
    data = COCO(source_directory.joinpath("jsons/ins_seg_train_cocofmt.json"))

    final_boxes = dict()
    n_instances = 0
    n_matched_boxes = 0
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        image_path = source_directory.joinpath("images", "10k", "train", image_info["file_name"])
        
        im = Image.open(image_path)
        im = _apply_exif_orientation(im)
        im = np.array(im.convert("RGB"))
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        if len(annotations) == 0:
            continue
        segmentation = []
        for annotation in annotations:
            if annotation["iscrowd"] or annotation["ignore"]:
                continue
            if annotation["category_id"] != 1:
                continue
            seg = data.annToMask(annotation)
            segmentation.append(seg)
        if len(segmentation) == 0:
            continue
        boxes = detector.detect(im)
        n_instances += len(segmentation)
        if len(boxes) == 0:
            continue

        segmentation = np.stack(segmentation)
        matches, filtered_boxes_by_IoU_thr = get_matches(boxes, segmentation)
        final_boxes_ = [boxes[i, :4].tolist() for i, j in matches]
        final_boxes[str(image_id)] = final_boxes_
        n_matched_boxes += len(final_boxes_)
        if len(filtered_boxes_by_IoU_thr) > 0:
            im = cv2.imread(str(image_path))
            boxes = [boxes[i, :4] for i in filtered_boxes_by_IoU_thr]
            draw_faces(im, boxes)
            out_dir = Path(f"tools/anonymization/bdd100k/filtered_by_IOU/{image_path.name}")
            out_dir.parent.mkdir(exist_ok=True, parents=True)
            print("SAved to", out_dir)
            mask = segmentation.any(axis=0)[:, :, None]
            im = (im*.7 + 0.3*mask*255).astype(np.uint8)
            cv2.imwrite(str(out_dir), im)
            #cv2.imshow("frame", im)
            #key = cv2.waitKey(0)
    print(f"Out of {n_instances} instances, {n_matched_boxes} are matched to a segmentation.")

    target_path = source_directory.joinpath("face_boxes_train.json")
    with open(target_path, "w") as fp:
        json.dump(final_boxes, fp)


if __name__ == "__main__":
    annotate_bdd100k()
