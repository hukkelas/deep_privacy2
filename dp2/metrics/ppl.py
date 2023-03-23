import numpy as np
import torch
import tops
from dp2 import utils
from torch_fidelity.helpers import get_kwarg, vassert
from torch_fidelity.defaults import DEFAULTS as PPL_DEFAULTS
from torch_fidelity.utils import sample_random, batch_interp, create_sample_similarity
from torchvision.transforms.functional import resize


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


@torch.no_grad()
def calculate_ppl(
        dataloader,
        generator,
        latent_space=None,
        data_len=None,
        upsample_size=None,
        **kwargs) -> dict:
    """
    Inspired by https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """
    if latent_space is None:
        latent_space = generator.latent_space
    assert latent_space in ["Z", "W"], f"Not supported latent space: {latent_space}"
    assert len(upsample_size) == 2
    epsilon = PPL_DEFAULTS["ppl_epsilon"]
    interp = PPL_DEFAULTS['ppl_z_interp_mode']
    similarity_name = PPL_DEFAULTS['ppl_sample_similarity']
    sample_similarity_resize = PPL_DEFAULTS['ppl_sample_similarity_resize']
    sample_similarity_dtype = PPL_DEFAULTS['ppl_sample_similarity_dtype']
    discard_percentile_lower = PPL_DEFAULTS['ppl_discard_percentile_lower']
    discard_percentile_higher = PPL_DEFAULTS['ppl_discard_percentile_higher']

    vassert(type(epsilon) is float and epsilon > 0, 'Epsilon must be a small positive floating point number')
    vassert(discard_percentile_lower is None or 0 < discard_percentile_lower < 100, 'Invalid percentile')
    vassert(discard_percentile_higher is None or 0 < discard_percentile_higher < 100, 'Invalid percentile')
    if discard_percentile_lower is not None and discard_percentile_higher is not None:
        vassert(0 < discard_percentile_lower < discard_percentile_higher < 100, 'Invalid percentiles')

    sample_similarity = create_sample_similarity(
        similarity_name,
        sample_similarity_resize=sample_similarity_resize,
        sample_similarity_dtype=sample_similarity_dtype,
        cuda=False,
        **kwargs
    )
    sample_similarity = tops.to_cuda(sample_similarity)
    rng = np.random.RandomState(get_kwarg('rng_seed', kwargs))
    distances = []
    if data_len is None:
        data_len = len(dataloader) * dataloader.batch_size
    z0 = sample_random(rng, (data_len, generator.z_channels), "normal")
    z1 = sample_random(rng, (data_len, generator.z_channels), "normal")
    if latent_space == "Z":
        z1 = batch_interp(z0, z1, epsilon, interp)
    print("Computing PPL IN", latent_space)
    distances = torch.zeros(data_len, dtype=torch.float32, device=tops.get_device())
    print(distances.shape)
    end = 0
    n_samples = 0
    for it, batch in enumerate(utils.tqdm_(dataloader, desc="Perceptual Path Length")):
        start = end
        end = start + batch["img"].shape[0]
        n_samples += batch["img"].shape[0]
        batch_lat_e0 = tops.to_cuda(z0[start:end])
        batch_lat_e1 = tops.to_cuda(z1[start:end])
        if latent_space == "W":
            w0 = generator.get_w(batch_lat_e0, update_emas=False)
            w1 = generator.get_w(batch_lat_e1, update_emas=False)
            w1 = w0.lerp(w1, epsilon)  # PPL end
            rgb1 = generator(**batch, w=w0)["img"]
            rgb2 = generator(**batch, w=w1)["img"]
        else:
            rgb1 = generator(**batch, z=batch_lat_e0)["img"]
            rgb2 = generator(**batch, z=batch_lat_e1)["img"]
        if rgb1.shape[-2] < upsample_size[0] or rgb1.shape[-1] < upsample_size[1]:
            rgb1 = resize(rgb1, upsample_size, antialias=True)
            rgb2 = resize(rgb2, upsample_size, antialias=True)
        rgb1 = utils.denormalize_img(rgb1).mul(255).byte()
        rgb2 = utils.denormalize_img(rgb2).mul(255).byte()

        sim = sample_similarity(rgb1, rgb2)
        dist_lat_e01 = sim / (epsilon ** 2)
        distances[start:end] = dist_lat_e01.view(-1)
    distances = distances[:n_samples]
    distances = tops.all_gather_uneven(distances).cpu().numpy()
    if tops.rank() != 0:
        return {"ppl/mean": -1, "ppl/std": -1}
    if tops.rank() == 0:
        cond, lo, hi = None, None, None
        if discard_percentile_lower is not None:
            lo = np.percentile(distances, discard_percentile_lower, interpolation='lower')
            cond = lo <= distances
        if discard_percentile_higher is not None:
            hi = np.percentile(distances, discard_percentile_higher, interpolation='higher')
            cond = np.logical_and(cond, distances <= hi)
        if cond is not None:
            distances = np.extract(cond, distances)
        return {
            "ppl/mean": float(np.mean(distances)),
            "ppl/std": float(np.std(distances)),
        }
    else:
        return {"ppl/mean"}
