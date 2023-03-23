import pathlib
from typing import Tuple
import numpy as np
import torch
import pathlib
try:
    import pyspng
    PYSPNG_IMPORTED = True
except ImportError:
    PYSPNG_IMPORTED = False
    print("Could not load pyspng. Defaulting to pillow image backend.")
    from PIL import Image
from tops import logger


class FDFDataset:

    def __init__(self,
                 dirpath,
                 imsize: Tuple[int],
                 load_keypoints: bool,
                 transform):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        self.transform = transform
        self.imsize = imsize[0]
        self.load_keypoints = load_keypoints
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        image_dir = self.dirpath.joinpath("images", str(self.imsize))
        self.image_paths = list(image_dir.glob("*.png"))
        assert len(self.image_paths) > 0,\
            f"Did not find images in: {image_dir}"
        self.image_paths.sort(key=lambda x: int(x.stem))
        self.landmarks = np.load(self.dirpath.joinpath("landmarks.npy")).reshape(-1, 7, 2).astype(np.float32)

        self.bounding_boxes = torch.load(self.dirpath.joinpath("bounding_box", f"{self.imsize}.torch"))
        assert len(self.image_paths) == len(self.bounding_boxes)
        assert len(self.image_paths) == len(self.landmarks)
        logger.log(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}, imsize={imsize}")

    def get_mask(self, idx):
        mask = torch.ones((1, self.imsize, self.imsize), dtype=torch.bool)
        bounding_box = self.bounding_boxes[idx]
        x0, y0, x1, y1 = bounding_box
        mask[:, y0:y1, x0:x1] = 0
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        if PYSPNG_IMPORTED:
            with open(impath, "rb") as fp:
                im = pyspng.load(fp.read())
        else:
            with Image.open(impath) as fp:
                im = np.array(fp)
        im = torch.from_numpy(np.rollaxis(im, -1, 0))
        masks = self.get_mask(index)
        landmark = self.landmarks[index]
        batch = {
            "img": im,
            "mask": masks,
        }
        if self.load_keypoints:
            batch["keypoints"] = landmark
        if self.transform is None:
            return batch
        return self.transform(batch)


class FDF256Dataset:

    def __init__(self,
                 dirpath,
                 load_keypoints: bool,
                 transform):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        self.transform = transform
        self.load_keypoints = load_keypoints
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        image_dir = self.dirpath.joinpath("images")
        self.image_paths = list(image_dir.glob("*.png"))
        assert len(self.image_paths) > 0,\
            f"Did not find images in: {image_dir}"
        self.image_paths.sort(key=lambda x: int(x.stem))
        self.landmarks = np.load(self.dirpath.joinpath("landmarks.npy")).reshape(-1, 7, 2).astype(np.float32)
        self.bounding_boxes = torch.from_numpy(np.load(self.dirpath.joinpath("bounding_box.npy")))
        assert len(self.image_paths) == len(self.bounding_boxes)
        assert len(self.image_paths) == len(self.landmarks)
        logger.log(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}")

    def get_mask(self, idx):
        mask = torch.ones((1, 256, 256), dtype=torch.bool)
        bounding_box = self.bounding_boxes[idx]
        x0, y0, x1, y1 = bounding_box
        mask[:, y0:y1, x0:x1] = 0
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        if PYSPNG_IMPORTED:
            with open(impath, "rb") as fp:
                im = pyspng.load(fp.read())
        else:
            with Image.open(impath) as fp:
                im = np.array(fp)
        im = torch.from_numpy(np.rollaxis(im, -1, 0))
        masks = self.get_mask(index)
        landmark = self.landmarks[index]
        batch = {
            "img": im,
            "mask": masks,
        }
        if self.load_keypoints:
            batch["keypoints"] = landmark
        if self.transform is None:
            return batch
        return self.transform(batch)
