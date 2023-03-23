import pathlib
import os
import torch
from tops.config import LazyCall as L

if "PRETRAINED_CHECKPOINTS_PATH" in os.environ:
    PRETRAINED_CHECKPOINTS_PATH = pathlib.Path(os.environ["PRETRAINED_CHECKPOINTS_PATH"])
else:
    PRETRAINED_CHECKPOINTS_PATH = pathlib.Path("pretrained_checkpoints")
if "BASE_OUTPUT_DIR" in os.environ:
    BASE_OUTPUT_DIR = pathlib.Path(os.environ["BASE_OUTPUT_DIR"])
else:
    BASE_OUTPUT_DIR = pathlib.Path("outputs")



common = dict(
    logger_backend=["wandb", "stdout", "json", "image_dumper"],
    wandb_project="deep_privacy2",
    output_dir=BASE_OUTPUT_DIR,
    experiment_name=None, # Optional experiment name to show on wandb
)

train = dict(
    batch_size=32,
    seed=0,
    ims_per_log=1024,
    ims_per_val=int(200e3),
    max_images_to_train=int(12e6),
    amp=dict(
        enabled=True,
        scaler_D=L(torch.cuda.amp.GradScaler)(init_scale=2**16, growth_factor=4, growth_interval=100, enabled="${..enabled}"),
        scaler_G=L(torch.cuda.amp.GradScaler)(init_scale=2**16, growth_factor=4, growth_interval=100, enabled="${..enabled}"),
    ),
    fp16_ddp_accumulate=False, # All gather gradients in fp16?
    broadcast_buffers=False,
    bias_act_plugin_enabled=True,
    grid_sample_gradfix_enabled=True,
    conv2d_gradfix_enabled=False,
    channels_last=False,
    compile_G=dict(
        enabled=False,
        mode="default" # default, reduce-overhead or max-autotune
    ),
    compile_D=dict(
        enabled=False,
        mode="default" # default, reduce-overhead or max-autotune
    )
)

# exponential moving average
EMA = dict(rampup=0.05)

