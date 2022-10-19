from pathlib import Path
from typing import Dict, List
import torchvision
import torch
import tops
import torchvision.transforms.functional as F
from .functional import hflip


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p: float,  flip_map=None,**kwargs):
        super().__init__()
        self.flip_ratio = p
        self.flip_map = flip_map
        if self.flip_ratio is None:
            self.flip_ratio = 0.5
        assert 0 <= self.flip_ratio <= 1

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1) > self.flip_ratio:
            return container
        return hflip(container, self.flip_map)


class CenterCrop(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, size: List[int]):
        super().__init__()
        self.size = tuple(size)

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        min_size = min(container["img"].shape[1], container["img"].shape[2])
        if min_size < self.size[0]:
            container["img"] = F.center_crop(container["img"], min_size)
            container["img"] = F.resize(container["img"], self.size)
            return container
        container["img"] = F.center_crop(container["img"], self.size)
        return container


class Resize(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = tuple(size)
        self.interpolation = interpolation

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = F.resize(container["img"], self.size, self.interpolation, antialias=True)
        if "semantic_mask" in container:
            container["semantic_mask"] = F.resize(
                container["semantic_mask"], self.size, F.InterpolationMode.NEAREST)
        if "embedding" in container:
            container["embedding"] = F.resize(
                container["embedding"], self.size, self.interpolation)
        if "mask" in container:
            container["mask"] = F.resize(
                container["mask"], self.size, F.InterpolationMode.NEAREST)
        if "E_mask" in container:
            container["E_mask"] = F.resize(
                container["E_mask"], self.size, F.InterpolationMode.NEAREST)
        if "maskrcnn_mask" in container:
            container["maskrcnn_mask"] = F.resize(
                container["maskrcnn_mask"], self.size, F.InterpolationMode.NEAREST)
        if "vertices" in container:
            container["vertices"] = F.resize(
                container["vertices"], self.size, F.InterpolationMode.NEAREST)
        return container

    def __repr__(self):
        repr = super().__repr__()
        vars_ = dict(size=self.size, interpolation=self.interpolation)
        return repr + " " +  " ".join([f"{k}: {v}" for k, v in vars_.items()])


class InsertHRImage(torch.nn.Module):
    """
    Resizes mask by maxpool and assumes condition is already created
    """
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = tuple(size)
        self.interpolation = interpolation

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert container["img"].dtype == torch.float32
        container["img_hr"] = F.resize(container["img"], self.size, self.interpolation, antialias=True)
        container["condition_hr"] = F.resize(container["condition"], self.size, self.interpolation, antialias=True)
        mask = container["mask"] > 0
        container["mask_hr"] = (torch.nn.functional.adaptive_max_pool2d(mask.logical_not().float(), output_size=self.size) > 0).logical_not().float()
        container["condition_hr"] = container["condition_hr"] * (1 - container["mask_hr"]) + container["img_hr"] * container["mask_hr"]
        return container

    def __repr__(self):
        repr = super().__repr__()
        vars_ = dict(size=self.size, interpolation=self.interpolation)
        return repr + " "


class Normalize(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, mean, std, inplace, keys=["img"]):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.keys = keys

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            container[key] = F.normalize(container[key], self.mean, self.std, self.inplace)
        return container

    def __repr__(self):
        repr = super().__repr__()
        vars_ = dict(mean=self.mean, std=self.std, inplace=self.inplace)
        return repr + " " + " ".join([f"{k}: {v}" for k, v in vars_.items()])


class ToFloat(torch.nn.Module):

    def __init__(self, keys=["img"], norm=True) -> None:
        super().__init__()
        self.keys = keys
        self.gain = 255 if norm else 1

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            container[key] = container[key].float() / self.gain
        return container


class RandomCrop(torchvision.transforms.RandomCrop):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = super().forward(container["img"])
        return container


class CreateCondition(torch.nn.Module):

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if container["img"].dtype == torch.uint8:
            container["condition"] = container["img"] * container["mask"].byte() + (1-container["mask"].byte()) * 127
            return container
        container["condition"] = container["img"] * container["mask"]
        return container


class CreateEmbedding(torch.nn.Module):

    def __init__(self, embed_path: Path, cuda=True) -> None:
        super().__init__()
        self.embed_map = torch.load(embed_path, map_location=torch.device("cpu"))
        if cuda:
            self.embed_map = tops.to_cuda(self.embed_map)

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vertices = container["vertices"]
        if vertices.ndim == 3:
            embedding = self.embed_map[vertices.long()].squeeze(dim=0)
            embedding = embedding.permute(2, 0, 1) * container["E_mask"]
            pass
        else:
            assert vertices.ndim == 4
            embedding = self.embed_map[vertices.long()].squeeze(dim=1) 
            embedding = embedding.permute(0, 3, 1, 2) * container["E_mask"]
        container["embedding"] = embedding
        container["embed_map"] = self.embed_map.clone()
        return container

