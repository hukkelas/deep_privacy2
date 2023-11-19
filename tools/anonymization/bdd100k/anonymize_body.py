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
from ..cityscapes.anonymize_body import cse_post_process_cfg
from dp2.detection.structures import PersonDetection
from dp2.anonymizer.histogram_match_anonymizers import HistogramMatchAnonymizer, LatentHistogramMatchAnonymizer

def iteratively_anonymize(target_directory: Path, anonymizer, synthesis_kwargs):

    source_directory = Path("/mnt/work2/haakohu/datasets/bdd100k")
    data = COCO(source_directory.joinpath("jsons/ins_seg_train_cocofmt.json"))

    with open(source_directory.joinpath("annotated_keypoints_train.json"), "r") as fp:
        import json
        all_keypoints = json.load(fp)
    vis_thr = .3
    all_boxes = []
    n_filtered = 0
    n_keypoints_all_zeros = 0
    n_instances = 0
    for image_id in tqdm.tqdm(data.imgs):
        image_info = data.loadImgs([image_id])[0]

        image_path = source_directory.joinpath("images", "10k", "train", image_info["file_name"])
        output_path = target_directory.joinpath(image_path.name)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        annotation_ids = data.getAnnIds([image_id])
        annotations = data.loadAnns(annotation_ids)
        if str(image_id) not in all_keypoints or len(annotations) == 0:
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
        
        segmentation = torch.from_numpy(np.stack(segmentation))
        keypoints = np.array(all_keypoints[str(image_id)]["keypoints"])
        keypoints = torch.from_numpy(np.stack(keypoints)).float()
        keypoints[:, :, 2] = keypoints[:, :, 2] > vis_thr
        n_instances += keypoints.shape[0]
        n_keypoints_all_zeros += (keypoints[:, :, -1] < 0.5).all(dim=-1).sum().item()
        im = Image.open(image_path)
        detection = PersonDetection(
            segmentation,
            **cse_post_process_cfg,
            orig_imshape_CHW=(3, *im.size[::-1]),
            keypoints=keypoints
        )
        
        detection.pre_process()
        all_boxes.extend(detection.boxes)
        n_filtered += len(keypoints) - len(detection)

        orig_im_mode = im.mode
        im = _apply_exif_orientation(im)
        im = im.convert("RGB")
        im = torch.from_numpy(np.rollaxis(np.array(im), 2))
        im = anonymizer.forward(im, detections=[detection], **synthesis_kwargs)
        im = Image.fromarray(utils.im2numpy(im)).convert(orig_im_mode)
        im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)
    print(f"Detected {n_instances}, where {n_keypoints_all_zeros} where not detected by keypoints. In total, {n_keypoints_all_zeros/n_instances*100:.2f}% where not detected with keypoints.")
    area = [((b[2] - b[0])*(b[3] - b[1]))**0.5 for b in all_boxes]
    widths = [(b[2] - b[0]) for b in all_boxes]
    heights = [(b[3] - b[1]) for b in all_boxes]
    print("Total boxes:", len(all_boxes))
    print("Number of boxes", len(all_boxes))
    print("Average area:", np.mean(area))
    print("Average width:", np.mean(widths))
    print("Average height:", np.mean(heights))
    print("Percentage over 288x160:", np.mean([a**2 > 288*160 for a in area]))
    print(f"Detected {n_instances}, where {n_filtered} where filtered by min size. In total, {n_filtered/n_instances*100:.2f}% where filtered out.")
    print("Num filtered boxes:", n_filtered)


@click.command()
@click.argument("config_path")
@click.option("--match-histogram", default=False, is_flag=True)
@click.option("--sampler", default="MMT", type=click.Choice(["MMT", "SMT", "None"]))
@click.option("--n-sampling-steps", "-n", default=1, type=int)
@click.option("--min_size", default=None, type=click.Choice([None, '16', '32']))
def main(config_path, match_histogram, sampler, n_sampling_steps, kp_fix, min_size):
    cfg = utils.load_config(config_path)
    target_dir = Path("/mnt/work2/haakohu/datasets/bdd100k_anonymized_2023/")
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
        name += f"_sample{n_sampling_steps}"

    if name not in ["dp2_body_gaussian", "dp2_body_maskout"]:
        name += f"_{sampler}"
    if min_size is not None:
        name += f"_minsize{min_size}"
        cse_post_process_cfg["exp_bbox_filter"]["minimum_area"] = int(min_size)*int(min_size)

    target_dir = target_dir.joinpath(name, "bdd100k", "images", "10k", "train")
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
        target_dir,
        anonymizer, synthesis_kwargs)

main()
