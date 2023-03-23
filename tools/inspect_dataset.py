import cv2
import torch
import click
import tops
from tops.config import LazyConfig, instantiate
from dp2 import utils
from dp2.utils import vis_utils
import numpy as np
from PIL import Image

def get_image(batch, cfg, fscale_vis):
    im0 = batch["condition"]
    im1 = batch["img"]
    im = utils.denormalize_img(torch.cat((im0, im1), dim=-1)).mul(255).byte()
    im = torch.cat((im, vis_utils.visualize_batch(**batch)), dim=-1)

    im = utils.im2numpy(im)

    im = tops.np_make_image_grid(im, nrow=len(im0))
    if fscale_vis != 1:
        new_shape = [int(_*fscale_vis) for _ in im.shape[:2][::-1]]
        im = np.array(Image.fromarray(im).resize(new_shape))
    return im


@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True)
@click.option("-n", "--num_images", default=8, type=int)
@click.option("-f", "--fscale_vis", default=1)
def main(config_path: str, train: bool, num_images: int, fscale_vis):
    cfg = LazyConfig.load(config_path)
    if train:
        dl_cfg = cfg.data.train.loader
    else:
        dl_cfg = cfg.data.val.loader
    dl_cfg.batch_size = num_images
    dl = instantiate(dl_cfg)
    print(dl.image_gpu_transform)
    dl = iter(dl)

    while True:
        batch = next(dl)
        im = get_image(batch, cfg, fscale_vis)
        cv2.imshow("", im[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()


if __name__ == "__main__":
    main()