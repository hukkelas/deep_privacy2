import cv2
import torch
import numpy as np
import click
import tops
import tqdm
from tops.config import instantiate
from PIL import Image
from dp2 import utils, infer
from dp2.utils import vis_utils
from torchvision.transforms.functional import resize


@torch.no_grad()
@torch.cuda.amp.autocast()
def get_im(dl, G, num_images, num_z, fscale_vis, truncation_value: float, b_idx, multi_modal_truncation, show_lowres: bool):
    ims = []
    for im_idx in tqdm.trange(num_images, desc="Sampling images"):
        batch = next(dl)
        ims.append(utils.im2numpy(batch["img"], True, True)[0])
        ims.append(utils.im2numpy(batch["condition"], True, True)[0])
        ims.append(utils.im2numpy(vis_utils.visualize_batch(**batch))[0])
        for z_idx in range(num_z):
            # Sample same Z by setting seed for different images
            tops.set_seed(b_idx*num_z + z_idx)
            if multi_modal_truncation and z_idx > 0:
                fake = G.multi_modal_truncate(**batch, truncation_value=0, w_indices=[z_idx-1])
            else:
                fake = G.sample(**batch, truncation_value=truncation_value)
            if "x_lowres" in fake and show_lowres:
                for x in fake["x_lowres"]:
                    x = resize(x, fake["img"].shape[-2:])
                    ims.append(utils.im2numpy(x, to_uint8=True, denormalize=True)[0])
            ims.append(utils.im2numpy(fake["img"], to_uint8=True, denormalize=True)[0])
    if fscale_vis != 1:
        new_shape = [int(_*fscale_vis) for _ in ims[0].shape[:2][::-1]]
        ims = [np.array(Image.fromarray(im).resize(new_shape)) for im in ims]
    im = tops.np_make_image_grid(ims, nrow=num_images)
    return im


@click.command()
@click.argument("config_path")
@click.option("-n", "--num_images", default=8)
@click.option("--num_z", "--nz", default=8)
@click.option("-f", "--fscale_vis", default=1, type=float, help="Scale the output image resultion")
@click.option("-t", "--truncation_value", default=None, type=float)
@click.option("-l", "--show-lowres", default=False, is_flag=True)
@click.option("--save", default=False, is_flag=True)
@click.option("--train", default=False, is_flag=True)
@click.option("--multi-modal-truncation", "--mt", default=False, is_flag=True)
def show_samples(
        config_path: str,
        save: bool,
        train: bool,
        **kwargs):
    tops.set_seed(1)
    cfg = utils.load_config(config_path)
    G = infer.build_trained_generator(cfg)
    cfg.train.batch_size = 1
    if train:
        dl_val = cfg.data.train.loader
    else:
        dl_val = cfg.data.val.loader
    dl_val.num_workers = 1
    dl_val.shuffle = False
    dl_val.infinite = False
    tops.set_seed(1)
    dl_val = iter(instantiate(dl_val))
    b_idx = 0
    im = get_im(dl_val, G, b_idx=b_idx, **kwargs)
    print("Press 'a' for next image, 'q' to quit.")
    while True:
        b_idx += 1
        cv2.imshow("image", im[:, :, ::-1])
        if save:
            cv2.imwrite("test.png", im[:, :, ::-1])
            print("Saved file to test.png")
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        if key == ord("a"):
            im = get_im(dl_val, G, b_idx=b_idx, **kwargs)

if __name__ == "__main__":
    show_samples()