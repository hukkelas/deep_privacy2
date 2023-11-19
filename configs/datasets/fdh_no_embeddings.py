
from dp2.data.transforms.transforms import ToFloat, Normalize, CreateCondition, InsertJointMap, RandomHorizontalFlip
from tops.config import LazyCall as L 
from dp2.data.utils import get_coco_flipmap
from tops.config import LazyCall as L 
import torch
from pathlib import Path
from .fdh import data, dataset_base_dir
data.imsize = (288, 160)
data_dir = Path(dataset_base_dir, "fdh_no_embeddings")
data.train.loader.update(
    path=data_dir.joinpath("train", "out-{000000..000612}.tar"),
    read_condition=False,
    gpu_transform=L(torch.nn.Sequential)(
        L(ToFloat)(keys=["img", "mask", "maskrcnn_mask"], norm=False),
        L(Normalize)(mean=[0.5*255, 0.5*255, 0.5*255], std=[0.5*255, 0.5*255, 0.5*255], inplace=True),
        L(CreateCondition)(),
    ),
    transform=L(torch.nn.Sequential)(
        L(RandomHorizontalFlip)(p=0.5, flip_map=get_coco_flipmap()),
        L(InsertJointMap)(imsize="${data.imsize}"),
    ),
    load_embedding=False,
    load_new_keypoints=True,
    num_workers=4,
)
data.val.loader.update(
    path=data_dir.joinpath("val", "out-{000000..000010}.tar"),
    num_workers=2,
    read_condition=False,
    load_embedding=False,
    transform=L(torch.nn.Sequential)(
       L(InsertJointMap)(imsize="${data.imsize}"),
    )
)
