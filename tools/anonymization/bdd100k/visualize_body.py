from PIL import Image
import numpy as np
import torch
import tqdm
from detectron2.data.detection_utils import _apply_exif_orientation
import pathlib
from dp2.detection.structures import PersonDetection
from pycocotools.coco import COCO
from ..cityscapes.anonymize_body import cse_post_process_cfg
from dp2 import utils

def main():
    source_directory = pathlib.Path("/mnt/work2/haakohu/datasets/bdd100k")
    data = COCO(source_directory.joinpath("jsons/ins_seg_train_cocofmt.json"))
    with open(source_directory.joinpath("annotated_keypoints_train.json"), "r") as fp:
        import json
        all_keypoints = json.load(fp)
    vis_thr = .5
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]

        image_path = source_directory.joinpath("images", "10k", "train", image_info["file_name"])
        im = Image.open(image_path)

        im = _apply_exif_orientation(im)
        im = np.array(im.convert("RGB"))
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        if str(image_id) not in all_keypoints or len(annotations) == 0:
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
        assert len(segmentation) > 0
        segmentation = torch.from_numpy(np.stack(segmentation))
        keypoints = np.array(all_keypoints[str(image_id)]["keypoints"])
        keypoints = torch.from_numpy(np.stack(keypoints)).float()
        keypoints[:, :, 2] = keypoints[:, :, 2] > vis_thr

        detection = PersonDetection(
            segmentation,
            **cse_post_process_cfg,
            orig_imshape_CHW=(3, *im.shape[:2]),
            keypoints=keypoints
        )
        im = torch.from_numpy(np.rollaxis(im, 2))
        im = detection.visualize(im)
        im = utils.im2numpy(im)
        import cv2
        cv2.imshow("lol", im[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()




main()
