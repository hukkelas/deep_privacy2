from .fdh18 import data, data_dir
data.imsize = (72, 40)
data.val.loader.update(
    path=data_dir.joinpath("val", "72", "out-{000000..000009}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "72", "out-{000000..000609}.tar"),
)