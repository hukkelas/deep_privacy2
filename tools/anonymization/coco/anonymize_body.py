import shutil
from PIL import Image
import numpy as np
import torch
import tqdm
from dp2 import utils
from detectron2.data.detection_utils import _apply_exif_orientation
from dp2.detection.structures import PersonDetection
import click, pathlib
from tops.config import instantiate
import tops
from pycocotools.coco import COCO
from dp2.anonymizer.histogram_match_anonymizers import HistogramMatchAnonymizer, LatentHistogramMatchAnonymizer


def iteratively_anonymize(
        source_directory: pathlib.Path,
        target_directory: pathlib.Path,
        anonymizer, synthesis_kwargs):
    target_directory.mkdir(exist_ok=True, parents=True)
    data = COCO(source_directory.joinpath("annotations", "person_keypoints_train2017.json"))
    vis_thr = .3
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]
        image_path = source_directory.joinpath("train2017", image_info["file_name"])
        
        im = Image.open(image_path)
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        segmentation = []
        keypoints = []
        output_path = target_directory.joinpath(image_path.name)
        for annotation in annotations:
            kp = np.array(annotation["keypoints"]).reshape(17, 3)
            if (kp[:, 2] == 0).all():
                continue
            seg = data.annToMask(annotation)
            segmentation.append(seg)
            keypoints.append(kp)
        if len(keypoints) == 0:
            shutil.copy(image_path, output_path)
            continue
        print(output_path)
        segmentation = torch.from_numpy(np.stack(segmentation))
        keypoints = torch.from_numpy(np.stack(keypoints)).float()
        keypoints[:,:, 2] = keypoints[:, :, 2] >= vis_thr
        cse_post_process_cfg = dict(
            target_imsize=(288, 160),
            exp_bbox_cfg=dict(percentage_background=0.3, axis_minimum_expansion=.1),
            exp_bbox_filter=dict(minimum_area=8*8, min_bbox_ratio_inside=0, aspect_ratio_range=[0, 99999]),
            dilation_percentage=0.02,
            kp_vis_thr=0.3
        )
        detection = PersonDetection(
            segmentation,
            **cse_post_process_cfg,
            orig_imshape_CHW=(3, *im.size[::-1]),
            keypoints=keypoints
        )
        im = torch.from_numpy(np.rollaxis(np.array(im), 2))
        im = anonymizer.forward(im, detections=[detection], **synthesis_kwargs)
        im = Image.fromarray(utils.im2numpy(im)).convert(orig_im_mode)
        
        im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)

@click.command()
@click.argument("config_path")
@click.option("--match-histogram", default=False, is_flag=True)
@click.option("--sampler", default="MMT", type=click.Choice(["MMT", "SMT", "None"]))
@click.option("--n-sampling-steps", "-n", default=1, type=int)
def main(config_path, match_histogram: bool, n_sampling_steps: int, sampler: str, kp_fix: bool,):
    cfg = utils.load_config(config_path)
    source_directory = pathlib.Path("/mnt/work2/haakohu/datasets/coco/")
    target_dir = pathlib.Path("/mnt/work2/haakohu/datasets/coco_anonymized_2023/")
    name = {
        "configs/anonymizers/traditional/body/gaussian.py": "dp2_body_gaussian",
        "configs/anonymizers/traditional/body/maskout.py": "dp2_body_maskout",
        "configs/anonymizers/FB_triagan.py": "dp2_body_triaGAN"
    }[config_path]
    if match_histogram:
        assert name not in ["dp2_body_gaussian", "dp2_body_maskout"]
        name += "_MH"
        cfg.anonymizer._target_ = HistogramMatchAnonymizer
    if n_sampling_steps > 1:
        cfg.anonymizer._target_ = LatentHistogramMatchAnonymizer

    if name not in ["dp2_body_gaussian", "dp2_body_maskout"]:
        name += f"_{sampler}"
    if n_sampling_steps > 1:
        name += f"_sample{n_sampling_steps}"
    target_dir = target_dir.joinpath(name, "coco", "train2017")
    tops.set_seed(0)
    tops.set_AMP(False)

    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(
        amp=False,
        multi_modal_truncation=sampler == "MMT",
        truncation_value=1 if sampler == "None" else 0)
    if n_sampling_steps > 1:
        synthesis_kwargs["n_sampling_steps"] = n_sampling_steps

    iteratively_anonymize(
        source_directory, target_dir,
        anonymizer, synthesis_kwargs)

main()
