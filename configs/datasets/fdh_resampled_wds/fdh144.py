from .fdh18 import data, data_dir
data.imsize = (144, 80)
data.val.loader.update(
    path=data_dir.joinpath("val", "144", "out-{000000..000009}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "144", "out-{000000..000609}.tar"),
)