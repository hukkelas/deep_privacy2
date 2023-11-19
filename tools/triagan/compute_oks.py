from pathlib import Path
import numpy as np
import torch
import tqdm
import cv2
import io
import click
from PIL import Image  # This fixes a bug in webdataset writer...
from dp2.detection.models.vit_pose.vit_pose import VitPoseModel
from dp2.infer import build_trained_generator
from tops.config import instantiate
from dp2 import utils
from dp2.detection.utils import masks_to_boxes


def computeOks(dt, gt, bbox_area):
    kpt_oks_sigmas = (
        np.array(
            [
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
            ]
        )
        / 10.0
    )
    sigmas = kpt_oks_sigmas
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    # create bounds for ignore regions(double the gt bbox)
    g = gt
    xg, yg, vg = gt.T
    # xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    d = dt
    xd, yd, vd = dt.T
    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        return None
        z = np.zeros((k))
        raise Exception()
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
    e = (dx**2 + dy**2) / vars / (bbox_area + np.spacing(1)) / 2
    e = e[vg > 0]
    iou = np.sum(np.exp(-e)) / e.shape[0]
    return iou


@click.command()
@click.argument("config_path")
@torch.no_grad()
def main(config_path):
    pose_model = VitPoseModel("vit_huge")
    cfg = utils.load_config(config_path)
    G = build_trained_generator(cfg)
    cfg.train.batch_size = 1
    dl = instantiate(cfg.data.val.loader)
    vis_thr = 0.5  # Same as used for annotating FDH
    oks = 0
    n = 0
    for i, batch in enumerate(tqdm.tqdm(iter(dl), total=30_000)):
        box = masks_to_boxes((batch["mask"][0] > 0.5).logical_not())
        keypoints = batch["keypoints"][0].clone()
        im = batch["img"]
        with torch.cuda.amp.autocast():
            fake = G.sample(**batch, truncation_value=0)["img"]
        keypoints[:, 0] *= im.shape[-1]
        keypoints[:, 1] *= im.shape[-2]
        img_uint8 = fake.add(1).div(2).mul(255).round().byte()[0]
        predicted_keypoints = pose_model(img_uint8, box)[0]

        assert keypoints.shape == predicted_keypoints.shape, (
            keypoints.shape,
            predicted_keypoints.shape,
        )
        keypoints[:, -1] = keypoints[:, -1] > vis_thr
        predicted_keypoints[:, -1] = predicted_keypoints[:, -1] > vis_thr
        assert predicted_keypoints.shape == (17, 3), predicted_keypoints.shape
        oks_ = computeOks(
            keypoints.cpu().numpy(),
            predicted_keypoints.cpu().numpy(),
            im.shape[-1] * im.shape[-2],
        )
        if oks_ is None:
            continue
        oks += oks_
        n += 1

    oks = oks / n
    print("FINAL OKS:", oks)


if __name__ == "__main__":
    main()
