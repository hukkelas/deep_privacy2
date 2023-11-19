from pathlib import Path
import numpy as np
import torch
import tqdm
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from PIL import Image
from detectron2.structures.masks import polygons_to_bitmask
from dp2.detection.structures import PersonDetection
from dp2 import utils
import cv2
from detectron2.data.detection_utils import _apply_exif_orientation

cse_post_process_cfg = dict(
    target_imsize=(288, 160),
    exp_bbox_cfg=dict(percentage_background=0.3, axis_minimum_expansion=.1),
    exp_bbox_filter=dict(minimum_area=8*8, min_bbox_ratio_inside=0, aspect_ratio_range=[0, 99999]),
    dilation_percentage=0.02,
    kp_vis_thr=0.3
)
ddir = Path("/mnt/work2/haakohu/datasets/cityscapes/")
def iterate_dataset(split="train"):
    dataset = load_cityscapes_instances(ddir.joinpath(f"leftImg8bit/{split}"), ddir.joinpath(f"gtFine/{split}"))
    vis_thr = 0.3
    with open(ddir.joinpath(f"annotated_keypoints_{split}.json"), "r") as fp:
        import json
        data = json.load(fp)
    n_instances = 0
    n_detected_with_keypoints = 0
    for sample in dataset: # dict_keys(['file_name', 'image_id', 'height', 'width', 'annotations'])
        keypoints = data[sample["image_id"]]
        boxes = []
        masks = []
        kp_to_keep = []
        cur_idx = 0
        for i, instance_annotation in enumerate(sample["annotations"]): # dict_keys(['iscrowd', 'category_id', 'segmentation', 'bbox', 'bbox_mode'])
            # Detectron2 filters out all instances part of crowd
            if int(instance_annotation["category_id"]) != 0:
                continue

            # In keypoint annotation I did not filter out crowd label
            assert isinstance(instance_annotation["iscrowd"], bool)
            if instance_annotation["iscrowd"]:
                cur_idx += 1
                continue
            n_instances += 1
            kp_to_keep.append(cur_idx)
            bbox_XYXY = instance_annotation["bbox"]
            mask = polygons_to_bitmask(instance_annotation["segmentation"], height=sample["height"], width=sample["width"])
            masks.append(mask)
            boxes.append(bbox_XYXY)
            cur_idx += 1

        if len(boxes) == 0:
            yield sample["file_name"], None, None
            continue

        keypoints = [keypoints[i] for i in kp_to_keep]
        n_detected_with_keypoints += len(keypoints)
        boxes = torch.from_numpy(np.stack(boxes))
        masks = torch.from_numpy(np.stack(masks))
        keypoints = torch.from_numpy(np.stack(keypoints))
        keypoints[:, :, -1] = keypoints[:, :, -1] >= vis_thr

        yield sample["file_name"], keypoints, masks
    print(f"There are {n_instances} where {n_detected_with_keypoints} are detected with keypoints")


def main():
    #print(dataset[0]["annotations"][0].keys(())
    for file_name, keypoints, masks in iterate_dataset():
        continue
        if keypoints is None:
            continue
        im = Image.open(file_name)
        im = _apply_exif_orientation(im)
        im = im.convert("RGB")
        detection = PersonDetection(masks, **cse_post_process_cfg, orig_imshape_CHW=(3, *im.size[::-1]), keypoints=keypoints)
        im = torch.from_numpy(np.rollaxis(np.array(im), 2))
        im = detection.visualize(im)
        im = utils.im2numpy(im)
        cv2.imshow("test", im[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()

if __name__ == "__main__":
    main()