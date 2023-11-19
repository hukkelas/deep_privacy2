from pathlib import Path
from typing import Dict, List
import torchvision
import torch
import tops
import torchvision.transforms.functional as F
from .functional import hflip
import numpy as np
from dp2.utils.vis_utils import get_coco_keypoints
from PIL import Image, ImageDraw
from typing import Tuple


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p: float,  flip_map=None, **kwargs):
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
        return repr + " " + " ".join([f"{k}: {v}" for k, v in vars_.items()])


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


class InsertJointMap(torch.nn.Module):

    def __init__(self, imsize: Tuple) -> None:
        super().__init__()
        self.imsize = imsize
        knames = get_coco_keypoints()[0]
        knames = knames + ["neck", "mid_hip"]
        connectivity = {
            "nose": ["left_eye", "right_eye", "neck"],
            "left_eye": ["right_eye", "left_ear"],
            "right_eye": ["right_ear"],
            "left_shoulder": ["right_shoulder", "left_elbow", "left_hip"],
            "right_shoulder": ["right_elbow", "right_hip"],
            "left_elbow": ["left_wrist"],
            "right_elbow": ["right_wrist"],
            "left_hip": ["right_hip", "left_knee"],
            "right_hip": ["right_knee"],
            "left_knee": ["left_ankle"],
            "right_knee": ["right_ankle"],
            "neck": ["mid_hip", "nose"],
        }
        category = {
            ("nose", "left_eye"): 0,  # head
            ("nose", "right_eye"): 0,  # head
            ("nose", "neck"): 0,  # head
            ("left_eye", "right_eye"): 0,  # head
            ("left_eye", "left_ear"): 0,  # head
            ("right_eye", "right_ear"): 0,  # head
            ("left_shoulder", "left_elbow"): 1,  # left arm
            ("left_elbow", "left_wrist"): 1,  # left arm
            ("right_shoulder", "right_elbow"): 2,  # right arm
            ("right_elbow", "right_wrist"): 2,  # right arm
            ("left_shoulder", "right_shoulder"): 3,  # body
            ("left_shoulder", "left_hip"): 3,  # body
            ("right_shoulder", "right_hip"): 3,  # body
            ("left_hip", "right_hip"): 3,  # body
            ("left_hip", "left_knee"): 4,  # left leg
            ("left_knee", "left_ankle"): 4,  # left leg
            ("right_hip", "right_knee"): 5,  # right leg
            ("right_knee", "right_ankle"): 5,  # right leg
            ("neck", "mid_hip"): 3,  # body
            ("neck", "nose"): 0,  # head
        }
        self.indices2category = {
            tuple([knames.index(n) for n in k]): v for k, v in category.items()
        }
        self.connectivity_indices = {
            knames.index(k): [knames.index(v_) for v_ in v]
            for k, v in connectivity.items()
        }
        self.l_shoulder = knames.index("left_shoulder")
        self.r_shoulder = knames.index("right_shoulder")
        self.l_hip = knames.index("left_hip")
        self.r_hip = knames.index("right_hip")
        self.l_eye = knames.index("left_eye")
        self.r_eye = knames.index("right_eye")
        self.nose = knames.index("nose")
        self.neck = knames.index("neck")

    def create_joint_map(self, N, H, W, keypoints):
        joint_maps = np.zeros((N, H, W), dtype=np.uint8)
        for bidx, keypoints in enumerate(keypoints):
            assert keypoints.shape == (17, 3), keypoints.shape
            keypoints = torch.cat((keypoints, torch.zeros(2, 3)))
            visible = keypoints[:, -1] > 0

            if visible[self.l_shoulder] and visible[self.r_shoulder]:
                neck = (keypoints[self.l_shoulder]
                        + (keypoints[self.r_shoulder] - keypoints[self.l_shoulder]) / 2)
                keypoints[-2] = neck
                visible[-2] = 1
            if visible[self.l_hip] and visible[self.r_hip]:
                mhip = (keypoints[self.l_hip]
                        + (keypoints[self.r_hip] - keypoints[self.l_hip]) / 2
                        )
                keypoints[-1] = mhip
                visible[-1] = 1

            keypoints[:, 0] *= W
            keypoints[:, 1] *= H
            joint_map = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
            draw = ImageDraw.Draw(joint_map)
            for fidx in self.connectivity_indices.keys():
                for tidx in self.connectivity_indices[fidx]:
                    if visible[fidx] == 0 or visible[tidx] == 0:
                        continue
                    c = self.indices2category[(fidx, tidx)]
                    s = tuple(keypoints[fidx, :2].round().long().numpy().tolist())
                    e = tuple(keypoints[tidx, :2].round().long().numpy().tolist())
                    draw.line((s, e), width=1, fill=c + 1)
            if visible[self.nose] == 0 and visible[self.neck] == 1:
                m_eye = (
                    keypoints[self.l_eye]
                    + (keypoints[self.r_eye] - keypoints[self.l_eye]) / 2
                )
                s = tuple(m_eye[:2].round().long().numpy().tolist())
                e = tuple(keypoints[self.neck, :2].round().long().numpy().tolist())
                c = self.indices2category[(self.nose, self.neck)]
                draw.line((s, e), width=1, fill=c + 1)
            joint_map = np.array(joint_map)

            joint_maps[bidx] = np.array(joint_map)
        return joint_maps[:, None]

    def forward(self, batch):
        batch["joint_map"] = torch.from_numpy(self.create_joint_map(
            batch["img"].shape[0], *self.imsize, batch["keypoints"]))
        batch["joint_map"] = batch["joint_map"].to(batch["img"].device)
        return batch
