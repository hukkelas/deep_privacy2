"""
Approach: "StyleMC: Multi-Channel Based Fast Text-Guided Image Generation and Manipulation"
Original source code: 
https://github.com/autonomousvision/stylegan_xl/blob/f9be58e98110bd946fcdadef2aac8345466faaf3/run_stylemc.py#
Modified by Håkon Hukkelås
"""

from pathlib import Path
import tqdm
from dp2 import utils
import tops
from timeit import default_timer as timer
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, normalize
import clip
from dp2.gan_trainer import AverageMeter


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def prompts_dist_loss(x, targets):
    loss = spherical_dist_loss
    if len(targets) == 1:
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)


@torch.no_grad()
def get_styles(seed, G: torch.nn.Module, batch, truncation_value=1):
    all_styles = []
    if seed is None:
        z = np.random.normal(0, 0, size=(1, G.z_channels))
    else:
        z = np.random.RandomState(seed=seed).normal(0, 1, size=(1, G.z_channels))
        z_idx = np.random.RandomState(seed=seed).randint(0, len(G.style_net.w_centers))
    w_c = G.style_net.w_centers[z_idx].to(tops.get_device()).view(1, -1)
    w = G.style_net(torch.from_numpy(z).to(tops.get_device()))

    w = w_c.to(w.dtype).lerp(w, truncation_value)
    if hasattr(G, "get_comod_y"):
        w = G.get_comod_y(batch, w)
    for block in G.modules():
        if not hasattr(block, "affine") or not hasattr(block.affine, "weight"):
            continue
        gamma0 = block.affine(w)
        if hasattr(block, "affine_beta"):
            beta0 = block.affine_beta(w)
            gamma0 = torch.cat((gamma0, beta0), dim=1)
        all_styles.append(gamma0)
    max_ch = max([s.shape[-1] for s in all_styles])
    all_styles = [F.pad(s, ((0, max_ch - s.shape[-1])), "constant", 0) for s in all_styles]
    all_styles = torch.cat(all_styles)
    return all_styles


def get_and_cache_direction(output_dir: Path, dl_val, G, text_prompt):
    cache_path = output_dir.joinpath(
        "stylemc_cache", text_prompt.replace(" ", "_") + ".torch")
    if cache_path.is_file():
        print("Loaded cache from:", cache_path)
        return torch.load(cache_path)
    direction = find_direction(G, text_prompt, dl_val=iter(dl_val))
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(direction, cache_path)
    return direction


@torch.cuda.amp.autocast()
def find_direction(
    G,
    text_prompt,
    n_iterations=128*8,
    batch_size=8,
    dl_val=None
):
    time_start = timer()
    clip_model = clip.load("ViT-B/16", device=tops.get_device())[0]
    target = [clip_model.encode_text(clip.tokenize(text_prompt).to(tops.get_device())).float()]
    first_batch = next(dl_val)
    first_batch["embedding"] = None if "embedding" not in first_batch else first_batch["embedding"]
    s = get_styles(0, G, first_batch)
    # stats tracker
    tracker = AverageMeter()
    n_iterations = n_iterations // batch_size

    # initalize styles direction
    direction = torch.zeros(s.shape, device=tops.get_device())
    direction.requires_grad_()
    utils.set_requires_grad(G, False)
    direction_tracker = torch.zeros_like(direction)
    opt = torch.optim.AdamW([direction], lr=0.05, betas=(0., 0.999), weight_decay=0.25)

    grads = []
    for seed_idx in tqdm.trange(n_iterations):
        # forward pass through synthesis network with new styles
        if seed_idx == 0:
            batch = first_batch
        else:
            batch = next(dl_val)
            batch["embedding"] = None if "embedding" not in batch else batch["embedding"]
        styles = get_styles(seed_idx, G, batch) + direction
        img = G(**batch, s=iter(styles))["img"]
        batch = {k: v.cpu() if v is not None else v for k, v in batch.items()}
        # clip loss
        img = (img + 1)/2
        img = normalize(img, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        img = resize(img, (224, 224))
        embeds = clip_model.encode_image(img)
        cos_sim = prompts_dist_loss(embeds, target)
        cos_sim.backward(retain_graph=True)
        # track stats
        tracker.update(dict(cos_sim=cos_sim, norm=torch.norm(direction)))
        if not (seed_idx % batch_size):
            opt.step()
            grads.append(direction.grad.clone())
            direction.grad.data.zero_()
            print(tracker.get_average())
            tracker = AverageMeter()

    # throw out fluctuating channels
    direction = direction.detach()
    direction[direction_tracker > n_iterations / 4] = 0
    print(direction)
    print(f"Time for direction search: {timer() - time_start:.2f} s")
    return direction
