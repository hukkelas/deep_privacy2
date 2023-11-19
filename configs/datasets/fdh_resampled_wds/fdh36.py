from .fdh18 import data, data_dir
data.imsize = (36, 20)
data.val.loader.update(
    path=data_dir.joinpath("val", "36", "out-{000000..000009}.tar"),
)
data.train.loader.update(
    path=data_dir.joinpath("train", "36", "out-{000000..000609}.tar"),
)