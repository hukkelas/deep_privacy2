import tempfile
import click
import tops
import warnings
import traceback
import torch
import os
from tops import checkpointer
from sg3_torch_utils.ops import conv2d_gradfix, grid_sample_gradfix, bias_act, upfirdn2d
from tops.config import instantiate
from tops import logger
from dp2 import utils, infer
from dp2.gan_trainer import GANTrainer


torch.backends.cudnn.benchmark = True


def start_train(rank, world_size, debug, cfg_path, temp_dir, benchmark: bool):
    print(rank, world_size)
    cfg = utils.load_config(cfg_path)
    if debug:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        conv2d_gradfix.enabled = cfg.train.conv2d_gradfix_enabled
        grid_sample_gradfix.enabled = cfg.train.grid_sample_gradfix_enabled
        upfirdn2d.enabled = cfg.train.grid_sample_gradfix_enabled
        bias_act.enabled = cfg.train.bias_act_plugin_enabled
    if world_size > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        init_method = f"file://{init_file}"
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size, init_method=init_method
        )
        # pin memory in dataloader would allocate memory on device:0 for distributed training.
        torch.cuda.set_device(tops.get_device())

    tops.set_AMP(cfg.train.amp.enabled)
    utils.init_tops(cfg)
    if tops.rank() == 0:
        utils.print_config(cfg)
        with open(cfg.output_dir.joinpath("config_path.py"), "w") as fp:
            fp.write(utils.config_to_str(cfg))

    if world_size > 1:
        assert cfg.train.batch_size > tops.world_size()
        assert cfg.train.batch_size % tops.world_size() == 0
        cfg.train.batch_size //= world_size
    if rank != 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    tops.set_seed(cfg.train.seed + rank)
    logger.log("Loading dataset.")
    dl_val = instantiate(cfg.data.val.loader, channels_last=cfg.train.channels_last)
    dl_train = instantiate(cfg.data.train.loader, channels_last=cfg.train.channels_last)
    dl_train = iter(dl_train)

    logger.log("Initializing models.")
    G = instantiate(cfg.generator)
    D = tops.to_cuda(instantiate(cfg.discriminator))
    if tops.rank() == 0:
        print(G)
        print(D)

    # TODO: EMA MIGHT NEED TO BE SYNCED ACCROSS GPUs before instantiate
    G_EMA = utils.EMA(G, cfg.train.batch_size * world_size, **cfg.EMA)
    G = tops.to_cuda(G)
    if world_size > 1:
        logger.log("Syncing models accross GPUs")
        # Distributed is implemented self. # Buffers are never broadcasted during training.
        for module in [G_EMA, G, D]:
            params_and_buffers = list(module.named_parameters())
            params_and_buffers += list(module.named_buffers())
            for name, param in params_and_buffers:
                torch.distributed.broadcast(param, src=0)
    if cfg.train.compile_D.enabled:
        compile_kwargs = instantiate(cfg.train.compile_D)
        compile_kwargs.pop("enabled")
        D = torch.compile(D, **compile_kwargs)
    if cfg.train.compile_G.enabled:
        compile_kwargs = instantiate(cfg.train.compile_G)
        compile_kwargs.pop("enabled")
        G = torch.compile(G, **compile_kwargs)
    logger.log("Initializing optimizers")
    grad_scaler_D = instantiate(cfg.train.amp.scaler_D)
    grad_scaler_G = instantiate(cfg.train.amp.scaler_G)

    G_optim = instantiate(cfg.G_optim, params=G.parameters())
    D_optim = instantiate(cfg.D_optim, params=D.parameters())

    loss_fnc = instantiate(cfg.loss_fnc, D=D, G=G)
    logger.add_scalar("stats/gpu_batch_size", cfg.train.batch_size)
    logger.add_scalar("stats/ngpus", world_size)

    D.train()
    G.train()
    if hasattr(cfg.train, "discriminator_init_cfg") and not benchmark:
        cfg_ = utils.load_config(cfg.train.discriminator_init_cfg)
        ckpt = checkpointer.load_checkpoint(cfg_.checkpoint_dir)["discriminator"]
        if hasattr(cfg_, "ckpt_mapper_D"):
            ckpt = instantiate(cfg_.ckpt_mapper_D)(ckpt)
        D.load_state_dict(ckpt)
    if hasattr(cfg.train, "generator_init_cfg") and not benchmark:
        cfg_ = utils.load_config(cfg.train.generator_init_cfg)
        ckpt = checkpointer.load_checkpoint(cfg_.checkpoint_dir)["EMA_generator"]
        if hasattr(cfg_, "ckpt_mapper"):
            ckpt = instantiate(cfg_.ckpt_mapper)(ckpt)
        infer.load_state_dict(G, ckpt)
        infer.load_state_dict(G_EMA.generator, ckpt)

    G_EMA.eval()
    if cfg.train.channels_last:
        G = G.to(memory_format=torch.channels_last)
        D = D.to(memory_format=torch.channels_last)

    if tops.world_size() > 1:
        torch.distributed.barrier()

    trainer = GANTrainer(
        G=G,
        D=D,
        G_EMA=G_EMA,
        D_optim=D_optim,
        G_optim=G_optim,
        dl_train=dl_train,
        dl_val=dl_val,
        scaler_D=grad_scaler_D,
        scaler_G=grad_scaler_G,
        ims_per_log=cfg.train.ims_per_log,
        max_images_to_train=cfg.train.max_images_to_train,
        ims_per_val=cfg.train.ims_per_val,
        loss_handler=loss_fnc,
        evaluate_fn=instantiate(cfg.data.train_evaluation_fn),
        batch_size=cfg.train.batch_size,
        broadcast_buffers=cfg.train.broadcast_buffers,
        fp16_ddp_accumulate=cfg.train.fp16_ddp_accumulate,
        save_state=not benchmark
    )
    if benchmark:
        trainer.estimate_ims_per_hour()
        if world_size > 1:
            torch.distributed.barrier()
        logger.finish()
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    try:
        trainer.train_loop()
    except Exception as e:
        traceback.print_exc()
        exit()
    tops.set_AMP(False)
    tops.set_seed(0)
    metrics = instantiate(cfg.data.evaluation_fn)(generator=G_EMA, dataloader=dl_val)
    metrics = {f"metrics_final/{k}": v for k, v in metrics.items()}
    logger.add_dict(metrics, level=logger.logger.INFO)
    if world_size > 1:
        torch.distributed.barrier()
    logger.finish()

    if world_size > 1:
        torch.distributed.destroy_process_group()


@click.command()
@click.argument("config_path")
@click.option("--debug", default=False, is_flag=True)
@click.option("--benchmark", default=False, is_flag=True)
def main(config_path: str, debug: bool, benchmark: bool):
    world_size = (
        torch.cuda.device_count()
    )  # Manually overriding this does not work. have to set CUDA_VISIBLE_DEVICES environment variable
    if world_size > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(
                start_train,
                args=(world_size, debug, config_path, temp_dir, benchmark),
                nprocs=torch.cuda.device_count(),
            )
    else:
        start_train(0, 1, debug, config_path, None, benchmark)

if __name__ == "__main__":
    main()
