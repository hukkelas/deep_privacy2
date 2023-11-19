import shutil
from PIL import Image
import numpy as np
import torch
import json
from dp2 import utils
from dp2.detection.structures import FaceDetection
import click, pathlib
from tops.config import instantiate
import tops
from .annotate_faces import iterate_dataset, ddir

def iteratively_anonymize(
        target_directory: pathlib.Path,
        anonymizer, synthesis_kwargs):
    target_directory.mkdir(exist_ok=True, parents=True)
    annotation_path = ddir.joinpath("face_boxes_train.json")
    with open(annotation_path, "r") as fp:
        annotation_boxes = json.load(fp)
    for impath, image_id, im, masks, orig_im_mode in iterate_dataset():
        file_name = "/".join(pathlib.Path(impath).parts[len(ddir.parts):])
        output_path = target_directory.joinpath(file_name)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if im is None or image_id not in annotation_boxes:
            shutil.copy(impath, output_path)
            continue
        boxes = torch.tensor(annotation_boxes[image_id]).round().long()
        if len(boxes) == 0:
            shutil.copy(impath, output_path)
            continue
        boxes = boxes[:, :4]
        detection = [
            FaceDetection(boxes, target_imsize=(128, 128), fdf128_expand=True)
        ]
        im = torch.from_numpy(np.rollaxis(im, 2))
        im = anonymizer(im, **synthesis_kwargs, detections=detection)
        im = utils.im2numpy(im)
        im = Image.fromarray(im).convert(orig_im_mode)
        im.save(output_path, optimize=False, quality=100, subsampling=0)


@click.command()
@click.argument("config_path")
def main(config_path):
    target_dir = pathlib.Path("/mnt/work2/haakohu/datasets/cityscapes_anonymized_2023/")
    name = {
        "configs/anonymizers/face_fdf128.py": "dp2_face_fdf128",
        "configs/anonymizers/traditional/face/gaussian.py": "dp2_face_gaussian",
        "configs/anonymizers/traditional/face/maskout.py": "dp2_face_maskout",
    }[config_path]
    target_dir = target_dir.joinpath(name, "cityscapes")
    tops.set_seed(0)
    tops.set_AMP(False)
    cfg = utils.load_config(config_path)
    cfg.detector.score_threshold = .3
    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(amp=False, multi_modal_truncation=False, truncation_value=0)
    iteratively_anonymize(
        target_dir,
        anonymizer, synthesis_kwargs)

main()
