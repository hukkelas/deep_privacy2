import tops
import tqdm
from tops import logger, highlight_py_str
from tops.config import LazyConfig


def print_config(cfg):
    logger.log("\n" + highlight_py_str(LazyConfig.to_py(cfg, prefix="")))


def config_to_str(cfg):
    return LazyConfig.to_py(cfg, prefix=".")


def init_tops(cfg, reinit=False):
    tops.init(
        cfg.output_dir, cfg.common.logger_backend,  cfg.experiment_name,
        cfg.common.wandb_project, dict(cfg), reinit)


def tqdm_(iterator, *args, **kwargs):
    if tops.rank() == 0:
        return tqdm.tqdm(iterator, *args, **kwargs)
    return iterator


def trange_(*args, **kwargs):
    if tops.rank() == 0:
        return tqdm.trange(*args, **kwargs)
    return range(*args)
