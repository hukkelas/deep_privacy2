import pathlib
from tops.config import LazyConfig
from .torch_utils import (
    im2torch, im2numpy, denormalize_img, set_requires_grad, forward_D_fake,
    binary_dilation, crop_box, remove_pad,
    torch_wasserstein_loss
)
from .ema import EMA
from .utils import init_tops, tqdm_, print_config, config_to_str, trange_
from .cse import from_E_to_vertex


def load_config(config_path):
    config_path = pathlib.Path(config_path)
    assert config_path.is_file(), config_path
    cfg = LazyConfig.load(str(config_path))
    cfg.output_dir = pathlib.Path(str(config_path).replace("configs", str(cfg.common.output_dir)).replace(".py", ""))
    if cfg.common.experiment_name is None:
        cfg.experiment_name = str(config_path)
    else:
        cfg.experiment_name = cfg.common.experiment_name
    cfg.checkpoint_dir = cfg.output_dir.joinpath("checkpoints")
    print("Saving outputs to:", cfg.output_dir)
    return cfg
