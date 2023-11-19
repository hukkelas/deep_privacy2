import tops
import torch
from tops import checkpointer
from tops.config import instantiate
from tops.logger import warn
from dp2.generator.deep_privacy1 import MSGGenerator


def load_generator_state(ckpt, G: torch.nn.Module, ckpt_mapper=None):
    state = ckpt["EMA_generator"] if "EMA_generator" in ckpt else ckpt["running_average_generator"]
    if ckpt_mapper is not None:
        state = ckpt_mapper(state)
    if isinstance(G, MSGGenerator):
        G.load_state_dict(state)
    else:
        load_state_dict(G, state)
    tops.logger.log(f"Generator loaded, num parameters: {tops.num_parameters(G)/1e6}M")
    if "w_centers" in ckpt:
        G.style_net.register_buffer("w_centers", ckpt["w_centers"])
        tops.logger.log(f"W cluster centers loaded. Number of centers: {len(G.style_net.w_centers)}")
    if "style_net.w_centers" in state:
        G.style_net.register_buffer("w_centers", state["style_net.w_centers"])
        tops.logger.log(f"W cluster centers loaded. Number of centers: {len(G.style_net.w_centers)}")


def build_trained_generator(cfg, map_location=None):
    map_location = map_location if map_location is not None else tops.get_device()
    G = instantiate(cfg.generator)
    G.eval()
    G.imsize = tuple(cfg.data.imsize) if hasattr(cfg, "data") else None
    if hasattr(cfg, "ckpt_mapper"):
        ckpt_mapper = instantiate(cfg.ckpt_mapper)
    else:
        ckpt_mapper = None
    if "model_url" in cfg.common:
        ckpt = tops.load_file_or_url(cfg.common.model_url, md5sum=cfg.common.model_md5sum)
        load_generator_state(ckpt, G, ckpt_mapper)
        return G.to(map_location)
    try:
        ckpt = checkpointer.load_checkpoint(cfg.checkpoint_dir, map_location="cpu")
        load_generator_state(ckpt, G, ckpt_mapper)
    except FileNotFoundError as e:
        tops.logger.warn(f"Did not find generator checkpoint in: {cfg.checkpoint_dir}")
    return G.to(map_location)


def build_trained_discriminator(cfg, map_location=None):
    map_location = map_location if map_location is not None else tops.get_device()
    D = instantiate(cfg.discriminator).to(map_location)
    D.eval()
    try:
        ckpt = checkpointer.load_checkpoint(cfg.checkpoint_dir, map_location="cpu")
        if hasattr(cfg, "ckpt_mapper_D"):
            ckpt["discriminator"] = instantiate(cfg.ckpt_mapper_D)(ckpt["discriminator"])
        D.load_state_dict(ckpt["discriminator"])
    except FileNotFoundError as e:
        tops.logger.warn(f"Did not find discriminator checkpoint in: {cfg.checkpoint_dir}")
    return D


def load_state_dict(module: torch.nn.Module, state_dict: dict):
    ignore_key = "style_net.w_centers" # Loaded by buyild_trained_generator
    module_sd = module.state_dict()
    to_remove = []
    for key, item in state_dict.items():
        if key not in module_sd:
            continue
        if item.shape != module_sd[key].shape:
            to_remove.append(key)
            warn(f"Incorrect shape. Current model: {module_sd[key].shape}, in state dict: {item.shape} for key: {key}")
    for key in to_remove:
        state_dict.pop(key)
    for key, item in state_dict.items():
        if key == ignore_key:
            continue
        if key not in module_sd:
            warn(f"Did not find key in model state dict: {key}")
    for key, item in module_sd.items():
        if key not in state_dict:
            warn(f"Did not find key in state dict: {key}")
    module.load_state_dict(state_dict, strict=False)
