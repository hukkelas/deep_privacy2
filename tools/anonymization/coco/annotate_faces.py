import numpy as np
import cv2
import tqdm
import json
from pathlib import Path
from face_detection import build_detector
from pycocotools.coco import COCO


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


def boxes2masks(boxes, shape):
    segmentation = np.zeros((len(boxes), *shape), dtype=bool)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(x) for x in box[:4]]
        segmentation[i, y0:y1, x0:x1] = 1
    return segmentation


def calculate_IoU(boxes, masks):
    boxes = boxes2masks(boxes, masks.shape[1:])
    masks = masks >= .5
    IoU = np.zeros((len(boxes), len(masks)), dtype=float)
    for i in range(len(boxes)):
        inter = np.logical_and(boxes[i:i+1], masks).reshape(masks.shape[0], -1).sum(axis=1)
        union = np.logical_or(boxes[i:i+1], masks).reshape(masks.shape[0], -1).sum(axis=1)
        IoU[i] = inter / (union +1e-7)
    return IoU

def get_matches(boxes: np.ndarray, masks: np.ndarray, min_iou_thr=0.01):
    IoU = calculate_IoU(boxes, masks)
    box_scores = boxes[:,  4]
    match_box_indices, match_mask_indices = np.where(IoU > min_iou_thr)
    match_mask_idx_unique, match_mask_idx_num_matches = np.unique(match_mask_indices, return_counts=True)
    mask_to_num_matches = np.zeros((len(masks)), dtype=int)
    mask_to_num_matches[match_mask_idx_unique] = match_mask_idx_num_matches
    # Sort by 
    match_indices = list(range(len(match_mask_indices)))
    def sort_key(match_idx):
        box_idx = match_box_indices[match_idx]
        mask_idx = match_mask_indices[match_idx]
        match_IoU = IoU[box_idx, mask_idx]
        num_matches = mask_to_num_matches[mask_idx]
        box_score = box_scores[box_idx]
        return (-match_IoU, -box_score, num_matches)
    match_indices.sort(key=sort_key)
    taken_box = np.zeros(((len(boxes))), dtype=bool)
    taken_mask = np.zeros((len(masks)), dtype=bool)
    final_matches = []
    for match_idx in match_indices:
        box_idx = match_box_indices[match_idx]
        mask_idx = match_mask_indices[match_idx]
        if taken_box[box_idx] or taken_mask[mask_idx]:
            continue
        taken_box[box_idx] = True
        taken_mask[mask_idx] = True
        final_matches.append((box_idx, mask_idx))
    
    # Get filtered boxes
    final_boxes = [i for i,j in final_matches]
    match_box_indices_, match_mask_indices_ = np.where(np.logical_and(IoU > 1e-7, IoU <= min_iou_thr))
    filtered_boxes_by_IoU_thr = [bidx for bidx, midx in zip(match_box_indices_, match_mask_indices_) if bidx not in match_box_indices and bidx not in final_boxes and not taken_mask[midx]]
    return final_matches, filtered_boxes_by_IoU_thr


def annotate_coco(coco_path: Path):
    data = COCO(coco_path.joinpath("annotations", "person_keypoints_train2017.json"))

    with open(coco_path.joinpath("initial_detected_boxes_train2017.json"), "r") as fp:
        face_detections = json.load(fp)
    final_boxes = dict()
    n_instances = 0
    n_matched_boxes = 0
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        image_path = coco_path.joinpath("train2017", image_info["file_name"])
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        if len(annotations) == 0:
            continue
        boxes = np.array(face_detections[str(image_id)]).reshape(-1, 5)

        segmentation = []
        keypoints = []
        for annotation in annotations:
            kp = np.array(annotation["keypoints"]).reshape(17, 3)
            if (kp[:, 2] == 0).all():
                continue
            seg = data.annToMask(annotation)
            segmentation.append(seg)
            keypoints.append(kp)
        n_instances += len(keypoints)
        if len(keypoints) == 0:
            continue
        if len(boxes) == 0:
            continue
        segmentation = np.stack(segmentation)
        matches, filtered_boxes_by_IoU_thr = get_matches(boxes, segmentation)
        final_boxes_ = [boxes[i, :4].tolist() for i, j in matches]
        keypoint_indices = [int(j) for i,j in matches]
        final_boxes[str(image_id)] = dict(boxes=final_boxes_, keypoint_indices=keypoint_indices)
        n_matched_boxes += len(final_boxes_)
        if len(filtered_boxes_by_IoU_thr) > 0:
            image_info = data.loadImgs([image_id])[0]
            image_path = coco_path.joinpath("train2017", image_info["file_name"])
            im = cv2.imread(str(image_path))
            boxes = [boxes[i, :4] for i in filtered_boxes_by_IoU_thr]
            draw_faces(im, boxes)
            out_dir = Path(f"tools/anonymization/coco/filtered_by_IOU/{image_path.name}")
            out_dir.parent.mkdir(exist_ok=True, parents=True)
            print("SAved to", out_dir)
            mask = segmentation.any(axis=0)[:, :, None]
            im = (im*.7 + 0.3*mask*255).astype(np.uint8)
            cv2.imwrite(str(out_dir), im)
            #cv2.imshow("frame", im)
            #key = cv2.waitKey(0)
    print(f"Out of {n_instances} instances, {n_matched_boxes} are matched to a segmentation.")

    target_path = coco_path.joinpath("boxes_train2017.json")
    with open(target_path, "w") as fp:
        json.dump(final_boxes, fp)


if __name__ == "__main__":
    detector = build_detector(
        clip_boxes=True,
        confidence_threshold=0.3
        )
    coco_path = Path("/mnt/work2/haakohu/datasets/coco/")
    annotate_coco(coco_path)
