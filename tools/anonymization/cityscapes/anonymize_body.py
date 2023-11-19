from .visualize_body import iterate_dataset, cse_post_process_cfg, ddir
from pathlib import Path
import numpy as np
import torch
import click
import tops
from tops.config import instantiate
from PIL import Image
from dp2.detection.structures import PersonDetection
from dp2 import utils
import shutil
from detectron2.data.detection_utils import _apply_exif_orientation
import tqdm
from dp2.anonymizer.histogram_match_anonymizers import HistogramMatchAnonymizer, LatentHistogramMatchAnonymizer

def iteratively_anonymize(anonymizer, target_directory: Path, synthesis_kwargs, split):
    target_directory.mkdir(exist_ok=True, parents=True)
    all_boxes = []
    n_keypoints_all_zeros = 0
    n_instances = 0
    n_filtered = 0
    for impath, keypoints, masks in tqdm.tqdm(iterate_dataset(split), total=2975 if split == "train" else 500):
        file_name = "/".join(Path(impath).parts[len(ddir.parts):])
        output_path = target_directory.joinpath(file_name)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if keypoints is None:
            shutil.copy(impath, output_path)
            continue
        im = Image.open(impath)
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode
        im = im.convert("RGB")
        n_instances += keypoints.shape[0]
        n_keypoints_all_zeros += (keypoints[:, :, -1] < 0.5).all(dim=-1).sum().item()
        detection = PersonDetection(masks, **cse_post_process_cfg, orig_imshape_CHW=(3, *im.size[::-1]), keypoints=keypoints.float())
        detection.pre_process()
        n_filtered += len(keypoints) - len(detection)

        all_boxes.extend(detection.boxes)
        im = torch.from_numpy(np.rollaxis(np.array(im), 2))
        im = anonymizer.forward(im, detections=[detection], **synthesis_kwargs)
        im = Image.fromarray(utils.im2numpy(im)).convert(orig_im_mode)
        print("Saving to:", output_path)
        im.save(output_path, optimize=False, quality=100, subsampling=0)
    area = [((b[2] - b[0])*(b[3] - b[1]))**0.5 for b in all_boxes]
    widths = [(b[2] - b[0]) for b in all_boxes]
    heights = [(b[3] - b[1]) for b in all_boxes]
    print(f"Detected {n_instances}, where {n_keypoints_all_zeros} where not detected by keypoints. In total, {n_keypoints_all_zeros/n_instances*100:.2f}% where not detected with keypoints.")
    print("Average area:", np.mean(area))
    print("Average width:", np.mean(widths))
    print("Average height:", np.mean(heights))
    print("Percentage over 288x160:", np.mean([a**2 > 288*160 for a in area]))
    print(f"Detected {n_instances}, where {n_filtered} where filtered by min size. In total, {n_filtered/n_instances*100:.2f}% where filtered out.")

@click.command()
@click.argument("config_path")
@click.option("--val", default=False, is_flag=True)
@click.option("--match-histogram", default=False, is_flag=True)
@click.option("--sampler", default="MMT", type=click.Choice(["MMT", "SMT", "None"]))
@click.option("--n-sampling-steps", "-n", default=1, type=int)
@click.option("--min_size", default=None, type=click.Choice([None, '16', '32']))
def main(config_path, val: bool, match_histogram, sampler, n_sampling_steps, kp_fix, min_size):
    cfg = utils.load_config(config_path)
    target_dir = Path("/mnt/work2/haakohu/datasets/cityscapes_anonymized_2023/")
    if val:
        target_dir = target_dir.parent.joinpath("cityscapes_anonymized_val")
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

    if min_size is not None:
        name += f"_minsize{min_size}"
        cse_post_process_cfg["exp_bbox_filter"]["minimum_area"] = int(min_size)*int(min_size)

    target_dir = target_dir.joinpath(name, "cityscapes")

    tops.set_seed(0)
    tops.set_AMP(False)

    anonymizer = instantiate(cfg.anonymizer, load_cache=True)
    synthesis_kwargs = dict(
        amp=False,
        multi_modal_truncation=sampler == "MMT",
        truncation_value=1 if sampler == "None" else 0,)
    if n_sampling_steps > 1:
        synthesis_kwargs["n_sampling_steps"] = n_sampling_steps
    iteratively_anonymize(anonymizer, target_dir, synthesis_kwargs, split="val" if val else "train")


if __name__ == "__main__":
    main()
