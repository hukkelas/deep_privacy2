from pathlib import Path
import torch
from tops.config import LazyCall as L
from dp2.data.transforms.transforms import Normalize, ToFloat
from ..fdh_no_embeddings import data, dataset_base_dir

data_dir = Path(dataset_base_dir, "fdh_no_embeddings_resampled/")
data.imsize = (18, 10)
data.val.loader.update(
    read_condition=True,
    path=data_dir.joinpath("val", "18", "out-{000000..000009}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "18", "out-{000000..000609}.tar"),
    read_condition=True,
    gpu_transform=L(torch.nn.Sequential)(
        L(ToFloat)(keys=["img", "mask", "maskrcnn_mask", "condition"], norm=False),
        L(Normalize)(mean=[0.5*255, 0.5*255, 0.5*255], std=[0.5*255, 0.5*255, 0.5*255], inplace=True, keys=["img", "condition"]),
    )
)
