import pickle
import numpy as np
import torch
import time
from pathlib import Path
from dp2 import utils
import tops
from .lpips import SampleSimilarityLPIPS
from torch_fidelity.defaults import DEFAULTS as trf_defaults
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.utils import create_feature_extractor
lpips_model = None
fid_model = None


@torch.no_grad()
def mse(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    se = (images1 - images2) ** 2
    se = se.view(images1.shape[0], -1).mean(dim=1)
    return se


@torch.no_grad()
def psnr(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    mse_ = mse(images1, images2)
    psnr = 10 * torch.log10(1 / mse_)
    return psnr


@torch.no_grad()
def lpips(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    return _lpips_w_grad(images1, images2)


def _lpips_w_grad(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    global lpips_model
    if lpips_model is None:
        lpips_model = tops.to_cuda(SampleSimilarityLPIPS())

    images1 = images1.mul(255)
    images2 = images2.mul(255)
    with torch.cuda.amp.autocast(tops.AMP()):
        dists = lpips_model(images1, images2)[0].view(-1)
    return dists


@torch.no_grad()
def compute_metrics_iteratively(
        dataloader, generator,
        cache_directory,
        data_len=None,
        truncation_value: float = None,
        multi_modal_truncate=False,
) -> dict:
    """
    Args:
        n_samples (int): Creates N samples from same image to calculate stats
        dataset_percentage (float): The percentage of the dataset to compute metrics on.
    """

    global lpips_model, fid_model
    if lpips_model is None:
        lpips_model = tops.to_cuda(SampleSimilarityLPIPS())
    if fid_model is None:
        fid_model = create_feature_extractor(
            trf_defaults["feature_extractor"], [trf_defaults["feature_layer_fid"]], cuda=False)
        fid_model = tops.to_cuda(fid_model)
    cache_directory = Path(cache_directory)
    start_time = time.time()
    lpips_total = torch.tensor(0, dtype=torch.float32, device=tops.get_device())
    diversity_total = torch.zeros_like(lpips_total)
    fid_cache_path = cache_directory.joinpath("fid_stats.pkl")
    has_fid_cache = fid_cache_path.is_file()
    if data_len is None:
        data_len = len(dataloader)*dataloader.batch_size
    if not has_fid_cache:
        fid_features_real = torch.zeros(data_len, 2048, dtype=torch.float32, device=tops.get_device())
    fid_features_fake = torch.zeros(data_len, 2048, dtype=torch.float32, device=tops.get_device())
    n_samples_seen = torch.tensor([0], dtype=torch.int32, device=tops.get_device())
    eidx = 0
    for batch in utils.tqdm_(iter(dataloader), desc="Computing FID, LPIPS and LPIPS Diversity"):
        sidx = eidx
        eidx = sidx + batch["img"].shape[0]
        n_samples_seen += batch["img"].shape[0]
        with torch.cuda.amp.autocast(tops.AMP()):
            if multi_modal_truncate:
                fakes1 = generator.multi_modal_truncate(n=batch["condition"].shape[0], truncation_value=0)["img"]
                fakes2 = generator.multi_modal_truncate(n=batch["condition"].shape[0], truncation_value=0)["img"]
            else:
                fakes1 = generator.sample(**batch, truncation_value=truncation_value)["img"]
                fakes2 = generator.sample(**batch, truncation_value=truncation_value)["img"]
            fakes1 = utils.denormalize_img(fakes1).mul(255)
            fakes2 = utils.denormalize_img(fakes2).mul(255)
            real_data = utils.denormalize_img(batch["img"]).mul(255)
            lpips_1, real_lpips_feats, fake1_lpips_feats = lpips_model(real_data, fakes1)
            fake2_lpips_feats = lpips_model.get_feats(fakes2)
            lpips_2 = lpips_model.lpips_from_feats(real_lpips_feats, fake2_lpips_feats)

            lpips_total += lpips_1.sum().add(lpips_2.sum()).div(2)
            diversity_total += lpips_model.lpips_from_feats(fake1_lpips_feats, fake2_lpips_feats).sum()
            if not has_fid_cache:
                fid_features_real[sidx:eidx] = fid_model(real_data.byte())[0]
            fid_features_fake[sidx:eidx] = fid_model(fakes1.byte())[0]
    fid_features_fake = fid_features_fake[:n_samples_seen]
    if has_fid_cache:
        if tops.rank() == 0:
            with open(fid_cache_path, "rb") as fp:
                fid_stat_real = pickle.load(fp)
    else:
        fid_features_real = fid_features_real[:n_samples_seen]
        fid_features_real = tops.all_gather_uneven(fid_features_real).cpu()
        if tops.rank() == 0:
            fid_stat_real = fid_features_to_statistics(fid_features_real)
            cache_directory.mkdir(exist_ok=True, parents=True)
            with open(fid_cache_path, "wb") as fp:
                pickle.dump(fid_stat_real, fp)
    fid_features_fake = tops.all_gather_uneven(fid_features_fake).cpu()
    if tops.rank() == 0:
        print("Starting calculation of fid from features of shape:", fid_features_fake.shape)
        fid_stat_fake = fid_features_to_statistics(fid_features_fake)
        fid_ = fid_statistics_to_metric(fid_stat_real, fid_stat_fake, verbose=False)["frechet_inception_distance"]
    tops.all_reduce(n_samples_seen, torch.distributed.ReduceOp.SUM)
    tops.all_reduce(lpips_total, torch.distributed.ReduceOp.SUM)
    tops.all_reduce(diversity_total, torch.distributed.ReduceOp.SUM)
    lpips_total = lpips_total / n_samples_seen
    diversity_total = diversity_total / n_samples_seen
    to_return = dict(lpips=lpips_total, lpips_diversity=diversity_total)
    if tops.rank() == 0:
        to_return["fid"] = fid_
    else:
        to_return["fid"] = -1
    to_return["validation_time_s"] = time.time() - start_time
    return to_return


@torch.no_grad()
def compute_lpips(
        dataloader, generator,
        truncation_value: float = None,
        data_len=None,
    ) -> dict:
    """
    Args:
        n_samples (int): Creates N samples from same image to calculate stats
        dataset_percentage (float): The percentage of the dataset to compute metrics on.
    """
    global lpips_model, fid_model
    if lpips_model is None:
        lpips_model = tops.to_cuda(SampleSimilarityLPIPS())
    start_time = time.time()
    lpips_total = torch.tensor(0, dtype=torch.float32, device=tops.get_device())
    diversity_total = torch.zeros_like(lpips_total)
    if data_len is None:
        data_len = len(dataloader) * dataloader.batch_size
    eidx = 0
    n_samples_seen = torch.tensor([0], dtype=torch.int32, device=tops.get_device())
    for batch in utils.tqdm_(dataloader, desc="Validating on dataset."):
        sidx = eidx
        eidx = sidx + batch["img"].shape[0]
        n_samples_seen += batch["img"].shape[0]
        with torch.cuda.amp.autocast(tops.AMP()):
            fakes1 = generator.sample(**batch, truncation_value=truncation_value)["img"]
            fakes2 = generator.sample(**batch, truncation_value=truncation_value)["img"]
            real_data = batch["img"]
            fakes1 = utils.denormalize_img(fakes1).mul(255)
            fakes2 = utils.denormalize_img(fakes2).mul(255)
            real_data = utils.denormalize_img(real_data).mul(255)
            lpips_1, real_lpips_feats, fake1_lpips_feats = lpips_model(real_data, fakes1)
            fake2_lpips_feats = lpips_model.get_feats(fakes2)
            lpips_2 = lpips_model.lpips_from_feats(real_lpips_feats, fake2_lpips_feats)

            lpips_total += lpips_1.sum().add(lpips_2.sum()).div(2)
            diversity_total += lpips_model.lpips_from_feats(fake1_lpips_feats, fake2_lpips_feats).sum()
    tops.all_reduce(n_samples_seen, torch.distributed.ReduceOp.SUM)
    tops.all_reduce(lpips_total, torch.distributed.ReduceOp.SUM)
    tops.all_reduce(diversity_total, torch.distributed.ReduceOp.SUM)
    lpips_total = lpips_total / n_samples_seen
    diversity_total = diversity_total / n_samples_seen
    to_return = dict(lpips=lpips_total, lpips_diversity=diversity_total)
    to_return = {k: v.cpu().item() for k, v in to_return.items()}
    to_return["validation_time_s"] = time.time() - start_time
    return to_return
