from .fdf128 import data, data_dir

data.imsize = (64, 64)
data.val.loader.update(
    path=data_dir.joinpath("val", "64", "out-{000000..000016}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "64", "out-{000000..000473}.tar"),
)
