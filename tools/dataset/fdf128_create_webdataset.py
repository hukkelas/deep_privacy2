from pathlib import Path
import webdataset as wds
import numpy as np
import torch
import tqdm
import io
from PIL import Image # This fixes a bug in webdataset writer...
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode
from dp2.data.datasets.fdf import FDFDataset

def png_decoder(x):
    with Image.open(io.BytesIO(x)) as im:
        return np.array(im)


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = np.load(io.BytesIO(x))
    return keypoints


def process_dataset(source_path: Path, target_path: Path, imsizes):
    dataset = FDFDataset(
        source_path, (128, 128), load_keypoints=True, transform=None
    )
    # max size is number of bytes
    target_paths = [
        target_path.joinpath(str(s[0]))
        for s in imsizes
    ]
    for t in target_paths:
        t.mkdir(exist_ok=True, parents=True)

    writers = [
        wds.ShardWriter(str(t.joinpath("out-%06d.tar")), maxsize=200*1024*1024, maxcount=3000) # maxsize 200MB
        for t in target_paths
    ]
    for i in tqdm.trange(len(dataset)):
        keypoints = dataset.landmarks[i]
        im = Image.open(dataset.image_paths[i])
        bounding_box = np.array(dataset.bounding_boxes[i]).astype(np.float32) / 128
        for widx, imsize in enumerate(imsizes):
            if imsize[0] != im.size[0]:
                im_ = np.array(im.resize(list(imsize)[::-1], resample=Image.BILINEAR))
            else:
                im_ = np.array(im)
            assert bounding_box.shape == (4, ), bounding_box.shape
            assert keypoints.shape == (7, 2), keypoints.shape
            writers[widx].write({
                "__key__": dataset.image_paths[i].stem,
                "image.png": im_,
                "keypoints.npy": keypoints,
                "bounding_box.npy": bounding_box
            })
    for w in writers:
        w.close()

def main():
    import os
    dataset_base_dir = (
        os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
    )
    data_dir = Path(dataset_base_dir, "fdf")
    sources = [
        data_dir.joinpath("val"),
        data_dir.joinpath("train"),
    ]
    target_path = [
        data_dir.parent.joinpath("fdf_resampled", "val"),
        data_dir.parent.joinpath("fdf_resampled", "train"),
    ]
    imsizes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128)
    ]
    for s, t in zip(sources, target_path):
        process_dataset(s, t, imsizes)
    

if __name__ == "__main__":
    main()