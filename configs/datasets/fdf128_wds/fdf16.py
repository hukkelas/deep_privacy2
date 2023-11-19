from .fdf128 import data, data_dir

data.imsize = (16, 16)
data.val.loader.update(
    path=data_dir.joinpath("val", "16", "out-{000000..000016}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "16", "out-{000000..000473}.tar"),
)
