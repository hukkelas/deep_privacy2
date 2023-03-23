import pickle
import torchvision
import torch
import pathlib
import numpy as np
from typing import Callable, Optional, Union
from torch.hub import get_dir as get_hub_dir


def cache_embed_stats(embed_map: torch.Tensor):
    mean = embed_map.mean(dim=0, keepdim=True)
    rstd = ((embed_map - mean).square().mean(dim=0, keepdim=True)+1e-8).rsqrt()

    cache = dict(mean=mean, rstd=rstd, embed_map=embed_map)
    path = pathlib.Path(get_hub_dir(), f"embed_map_stats.torch")
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(cache, path)


class CocoCSE(torch.utils.data.Dataset):

    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 normalize_E: bool,):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath

        self.transform = transform
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        self.image_paths, self.embedding_paths = self._load_impaths()
        self.embed_map = torch.from_numpy(np.load(self.dirpath.joinpath("embed_map.npy")))
        mean = self.embed_map.mean(dim=0, keepdim=True)
        rstd = ((self.embed_map - mean).square().mean(dim=0, keepdim=True)+1e-8).rsqrt()
        self.embed_map = (self.embed_map - mean) * rstd
        cache_embed_stats(self.embed_map)

    def _load_impaths(self):
        image_dir = self.dirpath.joinpath("images")
        image_paths = list(image_dir.glob("*.png"))
        image_paths.sort()
        embedding_paths = [
            self.dirpath.joinpath("embedding", x.stem + ".npy") for x in image_paths
        ]
        return image_paths, embedding_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im = torchvision.io.read_image(str(self.image_paths[idx]))
        vertices, mask, border = np.split(np.load(self.embedding_paths[idx]), 3, axis=-1)
        vertices = torch.from_numpy(vertices.squeeze()).long()
        mask = torch.from_numpy(mask.squeeze()).float()
        border = torch.from_numpy(border.squeeze()).float()
        E_mask = 1 - mask - border
        batch = {
            "img": im,
            "vertices": vertices[None],
            "mask": mask[None],
            "embed_map": self.embed_map,
            "border": border[None],
            "E_mask": E_mask[None]
        }
        if self.transform is None:
            return batch
        return self.transform(batch)
