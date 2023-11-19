import click
import torch
import os
import json
from dp2.infer import build_trained_generator
from tops.config import instantiate
from dp2.utils import load_config
import tops
import subprocess
from dp2.metrics.ppl import calculate_ppl
from dp2.metrics.fid_clip import compute_fid_clip
from dp2.metrics.torch_metrics import compute_metrics_iteratively

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def print_metric(path):
    with open(path, "r") as fp:
        all_metrics = json.load(fp)
    print("="*80)
    print("="*80)
    print("="*80)
    for k, v in all_metrics.items():
        print(f"{k:40}: {v}")
    print("="*80)
    print("="*80)
    print("="*80)

def validate(
        config_path,
        recompute: bool,
    ):
    tops.set_seed(0)
    tops.set_AMP(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = load_config(config_path)
    output_path = cfg.output_dir.joinpath("final_metrics.json")
    if output_path.is_file():
        print_metric(output_path)
        if not recompute:
            return

    batch_size = 8
    cfg.train.batch_size = batch_size   
    dl_val = instantiate(cfg.data.val.loader)
    G = build_trained_generator(cfg)
    tops.set_seed(0)

    ppl = calculate_ppl(dl_val, G, data_len=30_000, upsample_size=(288, 160))
    print(ppl)
    tops.set_seed(0)
    metrics = compute_metrics_iteratively(dl_val, G, cache_directory=f".final_metrics_cache{cfg.data.imsize[0]}", data_len=30_000)
    tops.set_seed(0)
    fid_clip = compute_fid_clip(dl_val, G, cache_directory=f".final_metrics_cache{cfg.data.imsize[0]}", data_len=30_000)
    commit = str(subprocess.check_output("git rev-parse HEAD", shell=True).decode()).strip()
    all_metrics = {
        **ppl, **metrics, **fid_clip,
    }
    all_metrics["commit"] = commit
    for k, v in all_metrics.items():
        if isinstance(v, torch.Tensor):
            print(k)
            all_metrics[k] = float(v.cpu().item())

    with open(output_path, mode="w") as fp:
        json.dump(all_metrics, fp)
    print_metric(output_path)



@click.command()
@click.argument("config_paths", nargs=-1)
@click.option("-r", "--recompute", default=False, is_flag=True)
def main(config_paths, recompute: bool):
    for cfg_path in config_paths:
        validate(cfg_path, recompute)


if __name__ == "__main__":
    main()
