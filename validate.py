import click
import torch
import os
import tempfile
from dp2.infer import build_trained_generator
from tops.config import instantiate
from dp2.utils import load_config
import tops
from tops import logger


def validate(
        rank,
        config_path,
        batch_size: int,
        truncation_value: float,
        multi_modal_truncate: bool,
        world_size,
        temp_dir,
        ):
    tops.set_seed(0)
    tops.set_AMP(False)
    if world_size > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size, init_method=init_method)
        torch.cuda.set_device(tops.get_device()) # pin memory in dataloader would allocate memory on device:0 for distributed training.
    cfg = load_config(config_path)

    if batch_size is not None:
        assert cfg.train.batch_size % world_size == 0
        cfg.train.batch_size = batch_size // world_size
    dl_val = instantiate(cfg.data.val.loader)
    G = build_trained_generator(cfg)
    tops.set_seed(0)
    tops.set_AMP(False)
    metrics = instantiate(cfg.data.evaluation_fn)(generator=G, dataloader=dl_val, truncation_value=truncation_value, multi_modal_truncate=multi_modal_truncate)
    metrics = {f"metrics_final/{k}": v for k,v in metrics.items()}
    if rank == 0:
        tops.init(cfg.output_dir)
        logger.add_dict(metrics)
        logger.finish()


@click.command()
@click.argument("config_path")
@click.option("--batch_size", default=16, type=int)
@click.option("--truncation-value", default=None, type=float)
@click.option("--multi-modal-truncate", "--mmt", default=False, is_flag=True)
def main(config_path, batch_size: int, truncation_value: float, multi_modal_truncate: bool):
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(validate,
                args=(config_path, batch_size, truncation_value, multi_modal_truncate, world_size, temp_dir),
                nprocs=world_size)
    else:
        validate(
            0, config_path, batch_size, truncation_value,multi_modal_truncate,
            world_size=1, temp_dir=None)


if __name__ == "__main__":
    main()