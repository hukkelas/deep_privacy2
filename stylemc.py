"""
Approach: "StyleMC: Multi-Channel Based Fast Text-Guided Image Generation and Manipulation"
Original source code: 
https://github.com/autonomousvision/stylegan_xl/blob/f9be58e98110bd946fcdadef2aac8345466faaf3/run_stylemc.py#
Modified by Håkon Hukkelås
"""
import click
from pathlib import Path
import tqdm
from dp2 import utils
import tops
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, normalize
import clip
from dp2.gan_trainer import AverageMeter
from tops.config import instantiate
from dp2.utils import vis_utils


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


affine_modules = None
max_ch = None


@torch.no_grad()
def init_affine_modules(G, batch):
    global affine_modules, max_ch
    affine_modules = []
    max_ch = 0

    def forward_hook(block, input_, output_):
        global max_ch
        affine_modules.append(block)
        max_ch = max(max_ch, block.affine.out_features * (1 + hasattr(block, "affine_beta")))
    removable_handles = []
    for block in G.modules():
        if hasattr(block, "affine") and hasattr(block.affine, "weight"):
            removable_handles.append(block.register_forward_hook(forward_hook))
    G(**batch)
    for hook in removable_handles:
        hook.remove()


@torch.no_grad()
def get_stylesW(w):
    global affine_modules, max_ch
    assert affine_modules is not None, "Have to run init_affine_modules first"

    all_styles = torch.zeros((len(affine_modules), max_ch), device=w.device, dtype=torch.float32)
    for i, block in enumerate(affine_modules):
        gamma0 = block.affine(w)
        if hasattr(block, "affine_beta"):
            beta0 = block.affine_beta(w)
            gamma0 = torch.cat((gamma0, beta0), dim=1)
        all_styles[i] = F.pad(gamma0, ((0, max_ch - gamma0.shape[-1])), "constant", 0)

    return all_styles

@torch.no_grad()
def get_styles(seed, G: torch.nn.Module, batch, truncation_value=1):
    global affine_modules, max_ch
    if affine_modules is None:
        init_affine_modules(G, batch)
    w = G.style_net.get_truncated(truncation_value, n=batch["condition"].shape[0], seed=seed)

    all_styles = torch.zeros((len(affine_modules), max_ch), device=batch["img"].device, dtype=torch.float32)
    for i, block in enumerate(affine_modules):
        gamma0 = block.affine(w)
        if hasattr(block, "affine_beta"):
            beta0 = block.affine_beta(w)
            gamma0 = torch.cat((gamma0, beta0), dim=1)
        all_styles[i] = F.pad(gamma0, ((0, max_ch - gamma0.shape[-1])), "constant", 0)

    return all_styles


def get_and_cache_direction(output_dir: Path, dl_val, G, text_prompt):
    cache_path = output_dir.joinpath("stylemc_cache", text_prompt.replace(" ", "_") + ".torch")
    if cache_path.is_file():
        return torch.load(cache_path)
    direction = find_direction(G, text_prompt, dl_val=iter(dl_val))
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(direction, cache_path)
    return direction

@torch.enable_grad()
@torch.cuda.amp.autocast()
def find_direction(
    G,
    text_prompt,
    n_iterations=128 * 8 * 10,
    batch_size=8,
    dl_val=None
):
    time_start = timer()
    clip_model = clip.load("ViT-B/16", device=tops.get_device())[0]
    target = [clip_model.encode_text(clip.tokenize(text_prompt).to(tops.get_device())).float()]
    first_batch = next(dl_val)
    first_batch["embedding"] = None if "embedding" not in first_batch else first_batch["embedding"]
    s = get_styles(0, G, first_batch, 0)
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

        # clip loss
        img = (img + 1) / 2
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


@click.command()
@click.argument("config_path")
@click.argument("text_prompt")
@click.option("-n", default=50, type=int)
def main(config_path: str, text_prompt: str, n: int):
    from dp2.infer import build_trained_generator
    from PIL import Image
    cfg = utils.load_config(config_path)
    G = build_trained_generator(cfg)
    cfg.train.batch_size = 1
    dl_val = instantiate(cfg.data.val.loader)
    direction = get_and_cache_direction(cfg.output_dir, dl_val, G, text_prompt)
    output_dir = Path("stylemc_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    strenghts = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0]
    for i, batch in enumerate(iter(dl_val)):
        imgs = []

        img = vis_utils.visualize_batch(**batch)
        img = tops.im2numpy(img, False)[0]
        imgs.append(img)
        if i > n:
            break
        for strength in strenghts:
            styles = get_styles(i, G, batch, truncation_value=0) + direction * strength
            img = G(**batch, s=iter(styles))["img"]
            imgs.append(utils.im2numpy(img, True, True)[0])

        img = tops.np_make_image_grid(imgs, nrow=1)
        Image.fromarray(img).save(output_dir.joinpath(f"results_{i}.png"))


if __name__ == "__main__":
    main()
