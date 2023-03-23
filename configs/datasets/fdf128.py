from pathlib import Path
from functools import partial
from dp2.data.datasets.fdf import FDFDataset
from .fdf256 import data, dataset_base_dir, metrics_cache, final_eval_fn, train_eval_fn

data_dir = Path(dataset_base_dir, "fdf")
data.train.dataset.dirpath = data_dir.joinpath("train")
data.val.dataset.dirpath = data_dir.joinpath("val")
data.imsize = (128, 128)
        

data.train_evaluation_fn = partial(
    train_eval_fn, cache_directory=Path(metrics_cache, "fdf128_val_train"))
data.evaluation_fn = partial(
    final_eval_fn, cache_directory=Path(metrics_cache, "fdf128_val_final"))

data.train.dataset.update(
    _target_ = FDFDataset,
    imsize="${data.imsize}"
)
data.val.dataset.update(
    _target_ = FDFDataset,
    imsize="${data.imsize}"
)