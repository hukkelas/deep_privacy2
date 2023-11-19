import pickle
import torch
import torchvision
from pathlib import Path
from dp2 import utils
import tops
try:
    import clip
except ImportError:
    print("Could not import clip.")
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
clip_model = None
clip_preprocess = None


@torch.no_grad()
def compute_fid_clip(
        dataloader, generator,
        cache_directory,
        data_len=None,
        truncation_value=None,
        multi_modal_truncate=False,
        **kwargs
    ) -> dict:
    """
    FID CLIP following the description in The Role of ImageNet Classes in Frechet Inception Distance, Thomas Kynkaamniemi et al.
    Args:
        n_samples (int): Creates N samples from same image to calculate stats
    """
    global clip_model, clip_preprocess
    if clip_model is None:
        clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
        normalize_fn = preprocess.transforms[-1]
        img_mean = normalize_fn.mean
        img_std = normalize_fn.std
        clip_model = tops.to_cuda(clip_model.visual)
        clip_preprocess = tops.to_cuda(torch.nn.Sequential(
            torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(img_mean, img_std)
        ))
    cache_directory = Path(cache_directory)
    if data_len is None:
        data_len = len(dataloader)*dataloader.batch_size
    fid_cache_path = cache_directory.joinpath("fid_stats_clip.pkl")
    has_fid_cache = fid_cache_path.is_file()
    if not has_fid_cache:
        fid_features_real = torch.zeros(data_len, 512, dtype=torch.float32, device=tops.get_device())
    fid_features_fake = torch.zeros(data_len, 512, dtype=torch.float32, device=tops.get_device())
    eidx = 0
    n_samples_seen = 0
    for batch in utils.tqdm_(iter(dataloader), desc="Computing FID CLIP."):
        sidx = eidx
        eidx = sidx + batch["img"].shape[0]
        n_samples_seen += batch["img"].shape[0]
        with torch.cuda.amp.autocast(tops.AMP()):
            if multi_modal_truncate:
                fakes = generator.multi_modal_truncate(n=batch["condition"].shape[0], truncation_value=0)["img"]
            else:
                fakes = generator.sample(n=batch["condition"].shape[0], truncation_value=truncation_value)["img"]
            real_data = batch["img"]
            fakes = utils.denormalize_img(fakes)
            real_data = utils.denormalize_img(real_data)
            if not has_fid_cache:
                real_data = clip_preprocess(real_data)
                fid_features_real[sidx:eidx] = clip_model(real_data)
            fakes = clip_preprocess(fakes)
            fid_features_fake[sidx:eidx] = clip_model(fakes)
    fid_features_fake = fid_features_fake[:n_samples_seen]
    fid_features_fake = tops.all_gather_uneven(fid_features_fake).cpu()
    if has_fid_cache:
        if tops.rank() == 0:
            with open(fid_cache_path, "rb") as fp:
                fid_stat_real = pickle.load(fp)
    else:
        fid_features_real = fid_features_real[:n_samples_seen]
        fid_features_real = tops.all_gather_uneven(fid_features_real).cpu()
        assert fid_features_real.shape == fid_features_fake.shape
        if tops.rank() == 0:
            fid_stat_real = fid_features_to_statistics(fid_features_real)
            cache_directory.mkdir(exist_ok=True, parents=True)
            with open(fid_cache_path, "wb") as fp:
                pickle.dump(fid_stat_real, fp)

    if tops.rank() == 0:
        print("Starting calculation of fid from features of shape:", fid_features_fake.shape)
        fid_stat_fake = fid_features_to_statistics(fid_features_fake)
        fid_ = fid_statistics_to_metric(fid_stat_real, fid_stat_fake, verbose=False)["frechet_inception_distance"]
        return dict(fid_clip=fid_)
    return dict(fid_clip=-1)
