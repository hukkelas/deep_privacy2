import shutil
from PIL import Image
import numpy as np
import torch
import json
import tqdm
from dp2 import utils
from detectron2.data.detection_utils import _apply_exif_orientation
from dp2.detection.structures import FaceDetection
import click, pathlib
from tops.config import instantiate
import tops
from pycocotools.coco import COCO

def iteratively_anonymize(
        anonymizer,
        target_directory: pathlib.Path,
        synthesis_kwargs):
    target_directory.mkdir(exist_ok=True, parents=True)
    coco_path = pathlib.Path("/mnt/work2/haakohu/datasets/coco/")
    data = COCO(coco_path.joinpath("annotations", "person_keypoints_train2017.json"))

    with open(coco_path.joinpath("boxes_train2017.json"), "r") as fp:
        face_detections = json.load(fp)
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        image_path = coco_path.joinpath("train2017", image_info["file_name"])
        output_path = target_directory.joinpath(image_path.name)
        if str(image_id) not in face_detections:
            shutil.copy(image_path, output_path)
            continue
        detections = face_detections[str(image_id)]
        boxes = torch.tensor(detections["boxes"]).round().long().view(-1, 4)
        if len(boxes) == 0:
            shutil.copy(image_path, output_path)
            continue
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        keypoints = []
        for annotation in annotations:
            kp = np.array(annotation["keypoints"]).reshape(17, 3)
            if (kp[:, 2] == 0).all():
                continue
            keypoints.append(kp)
        keypoints = torch.from_numpy(np.stack(keypoints)).float()
        keypoints = keypoints[detections["keypoint_indices"]]
        keypoints[:,:, 2] = keypoints[:, :, 2] > .5
        keypoints = keypoints[:, :, :2]
        assert len(keypoints) == len(boxes), (len(boxes), len(keypoints))
        im = Image.open(image_path)
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        im = np.array(im)
        detection = [
            FaceDetection(boxes, target_imsize=(128, 128), fdf128_expand=True, keypoints=keypoints)
        ]
        im = torch.from_numpy(np.rollaxis(im, 2))
        im = anonymizer(im, **synthesis_kwargs, detections=detection)
        im = utils.im2numpy(im)
        im = Image.fromarray(im).convert(orig_im_mode)
        im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)


@click.command()
@click.argument("config_path")
def main(config_path):
    target_dir = pathlib.Path("/mnt/work2/haakohu/datasets/coco_anonymized_2023/")
    name = {
        "configs/anonymizers/face_fdf128.py": "dp2_face_fdf128",
        "configs/anonymizers/deep_privacy1.py": "dp2_face_deep_privacy1",
        "configs/anonymizers/traditional/face/gaussian.py": "dp2_face_gaussian",
        "configs/anonymizers/traditional/face/maskout.py": "dp2_face_maskout",
    }[config_path]
    target_dir = target_dir.joinpath(name, "coco", "train2017")
    tops.set_seed(0)
    tops.set_AMP(False)
    cfg = utils.load_config(config_path)
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(amp=False, multi_modal_truncation=False, truncation_value=0)
    iteratively_anonymize(anonymizer, target_dir, synthesis_kwargs)

main()
