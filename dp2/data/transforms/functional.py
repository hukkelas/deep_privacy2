import torchvision.transforms.functional as F
import torch
import pickle
from tops import download_file, assert_shape
from typing import Dict
from functools import lru_cache

global symmetry_transform


@lru_cache(maxsize=1)
def get_symmetry_transform(symmetry_url):
    file_name = download_file(symmetry_url)
    with open(file_name, "rb") as fp:
        symmetry = pickle.load(fp)
    return torch.from_numpy(symmetry["vertex_transforms"]).long()


hflip_handled_cases = set([
    "keypoints", "img", "mask", "border", "semantic_mask", "vertices", "E_mask", "embed_map", "condition",
    "embedding", "vertx2cat", "maskrcnn_mask", "__key__"])


def hflip(container: Dict[str, torch.Tensor], flip_map=None) -> Dict[str, torch.Tensor]:
    container["img"] = F.hflip(container["img"])
    if "condition" in container:
        container["condition"] = F.hflip(container["condition"])
    if "embedding" in container:
        container["embedding"] = F.hflip(container["embedding"])
    assert all([key in hflip_handled_cases for key in container]), container.keys()
    if "keypoints" in container:
        assert flip_map is not None
        if container["keypoints"].ndim == 3:
            keypoints = container["keypoints"][:, flip_map, :]
            keypoints[:, :,  0] = 1 - keypoints[:, :,  0]
        else:
            assert_shape(container["keypoints"], (None, 3))
            keypoints = container["keypoints"][flip_map, :]
            keypoints[:, 0] = 1 - keypoints[:, 0]
        container["keypoints"] = keypoints
    if "mask" in container:
        container["mask"] = F.hflip(container["mask"])
    if "border" in container:
        container["border"] = F.hflip(container["border"])
    if "semantic_mask" in container:
        container["semantic_mask"] = F.hflip(container["semantic_mask"])
    if "vertices" in container:
        symmetry_transform = get_symmetry_transform(
            "https://dl.fbaipublicfiles.com/densepose/meshes/symmetry/symmetry_smpl_27554.pkl")
        container["vertices"] = F.hflip(container["vertices"])
        symmetry_transform_ = symmetry_transform.to(container["vertices"].device)
        container["vertices"] = symmetry_transform_[container["vertices"].long()]
    if "E_mask" in container:
        container["E_mask"] = F.hflip(container["E_mask"])
    if "maskrcnn_mask" in container:
        container["maskrcnn_mask"] = F.hflip(container["maskrcnn_mask"])
    return container
