import numpy as np
import tqdm
import click
from pathlib import Path
import tops
from pycocotools.coco import COCO
from PIL import Image
from detectron2.data.detection_utils import _apply_exif_orientation
from dp2 import utils
from tops.config import instantiate
import torch
import shutil
from dp2.detection.structures import FaceDetection

def iteratively_anonymize(target_directory: Path, anonymizer, synthesis_kwargs):

    source_directory = Path("/mnt/work2/haakohu/datasets/bdd100k")
    data = COCO(source_directory.joinpath("jsons/ins_seg_train_cocofmt.json"))

    with open(source_directory.joinpath("face_boxes_train.json"), "r") as fp:
        import json
        face_detections = json.load(fp)
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]

        image_path = source_directory.joinpath("images", "10k", "train", image_info["file_name"])
        output_path = target_directory.joinpath(image_path.name)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)

        if str(image_id) not in face_detections or len(annotations) == 0:
            shutil.copy(image_path, output_path)
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
            shutil.copy(image_path, output_path)
            continue

        assert len(segmentation) > 0
        boxes = torch.tensor(face_detections[str(image_id)]).round().long().view(-1, 4)
        detection = [
            FaceDetection(boxes, target_imsize=(128, 128), fdf128_expand=True)
        ]
        im = Image.open(image_path)
        orig_im_mode = im.mode
        im = _apply_exif_orientation(im)
        im = im.convert("RGB")
        im = torch.from_numpy(np.rollaxis(np.array(im), 2))
        im = anonymizer.forward(im, detections=detection, **synthesis_kwargs)
        im = Image.fromarray(utils.im2numpy(im)).convert(orig_im_mode)
        im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)


@click.command()
@click.argument("config_path")
def main(config_path):

    target_dir = Path("/mnt/work2/haakohu/datasets/bdd100k_anonymized_2023/")
    name = {
        "configs/anonymizers/traditional/face/gaussian.py": "dp2_face_gaussian",
        "configs/anonymizers/traditional/face/maskout.py": "dp2_face_maskout",
        "configs/anonymizers/face_fdf128.py": "dp2_face_fdf128",
    }[config_path]
    target_dir = target_dir.joinpath(name, "bdd100k", "images", "10k", "train")
    tops.set_seed(0)
    tops.set_AMP(False)
    cfg = utils.load_config(config_path)
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(amp=False, multi_modal_truncation=True, truncation_value=0)
    iteratively_anonymize(
        target_dir,
        anonymizer, synthesis_kwargs)


main()
