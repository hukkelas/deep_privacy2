import shutil
from PIL import Image
import numpy as np
import torch
import json
import tqdm
from dp2 import utils
from detectron2.data.detection_utils import _apply_exif_orientation
from dp2.detection.structures import FaceDetection
import pathlib
from tops.config import instantiate
import tops
from pycocotools.coco import COCO
from dp2.detection.box_utils_fdf import expand_bbox
def iteratively_anonymize(
        anonymizer,
        target_directory: pathlib.Path,
        synthesis_kwargs):
    criteria = 128*128
    target_directory.mkdir(exist_ok=True, parents=True)
    coco_path = pathlib.Path("/mnt/work2/haakohu/datasets/coco/")
    data = COCO(coco_path.joinpath("annotations", "person_keypoints_train2017.json"))
    image_dir = pathlib.Path("/mnt/work2/haakohu/datasets/coco_anonymized_2023/dp2_face_fdf128/coco/train2017")
    with open(coco_path.joinpath("boxes_train2017.json"), "r") as fp:
        face_detections = json.load(fp)
    num_boxes_total = 0
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        image_path = image_dir.joinpath(image_info["file_name"])
        output_path = target_directory.joinpath(image_path.name)
        if str(image_id) not in face_detections:
            shutil.copy(image_path, output_path)
            continue
        boxes = torch.tensor(face_detections[str(image_id)]["boxes"]).round().long().view(-1, 4)
        if len(boxes) == 0:
            shutil.copy(image_path, output_path)
            continue
        im = Image.open(image_path)
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        im = np.array(im)
        expanded_boxes_old = np.stack([expand_bbox(box, im.shape[-2:], False) for box in boxes.numpy()])
        area = (expanded_boxes_old[:, 2] - expanded_boxes_old[:, 0]) * (expanded_boxes_old[:, 3] - expanded_boxes_old[:, 1])
        boxes = boxes[area >= criteria]
        
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if len(boxes) == 0:
            shutil.copy(image_path, output_path)
            continue
        num_boxes_total += boxes.shape[0]
        detection = [
            FaceDetection(boxes, target_imsize=(256, 256), fdf128_expand=False)
        ]
        im = torch.from_numpy(np.rollaxis(im, 2))
        im = anonymizer(im, **synthesis_kwargs, detections=detection)
        im = utils.im2numpy(im)
        im = Image.fromarray(im).convert(orig_im_mode)
        im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)
    print("Total number of boxes larger than 128x128:", num_boxes_total)

def main():
    config_path = "configs/anonymizers/face.py"
    name = "dp2_face_fdf128_plus256_refine"
    target_dir = pathlib.Path("/mnt/work2/haakohu/datasets/coco_anonymized_2023/")
    target_dir = target_dir.joinpath(name, "coco", "train2017")
    tops.set_seed(0)
    tops.set_AMP(False)
    cfg = utils.load_config(config_path)
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(amp=False, multi_modal_truncation=False, truncation_value=0, n_sampling_steps=1)
    iteratively_anonymize(anonymizer, target_dir, synthesis_kwargs)

main()
