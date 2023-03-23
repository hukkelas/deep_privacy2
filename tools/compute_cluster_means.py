# Compute k-means cluster for W (Self-Distilled StyleGAN: Towards Generation from Internet Photos)
# pip install fast-pytorch-kmeans
import click
import tqdm
import torch
from dp2.utils import load_config
from dp2.infer import build_trained_generator
import tops
from tops.checkpointer.checkpointer import get_ckpt_paths, load_checkpoint
from fast_pytorch_kmeans import KMeans


@click.command()
@click.argument("config_path")
@click.option("-n", "--n_samples", default=int(600e3), type=int)
@click.option( "--n_centers", "--nc", default=512, type=int)
@click.option( "--batch_size", default=512, type=int)
def compute_cluster_means(config_path, n_samples, n_centers, batch_size):
    cfg = load_config(config_path)
    G = build_trained_generator(cfg, map_location=torch.device("cpu"))
    n_batches = n_samples // batch_size
    n_samples = n_samples // batch_size * batch_size
    print("Computing clusters over", n_samples, "samples.")
    style_net = G.stylenet if hasattr(G, "stylenet") else G.style_net
    style_net = tops.to_cuda(style_net)
    w_dim = style_net.w_dim
    z_dim = style_net.z_dim
    with torch.inference_mode():
        w = torch.zeros((n_samples, w_dim), device=tops.get_device(), dtype=torch.float32)

        for i in tqdm.trange(n_batches):
            w[i*batch_size:(i+1)*batch_size] = style_net(torch.randn((batch_size, z_dim), device=tops.get_device())).cpu()
        kmeans = KMeans(n_clusters=n_centers, mode='euclidean', verbose=10, max_iter=1000, tol=0.00001)

        kmeans.fit_predict(w)
        centers = kmeans.centroids

    if hasattr(style_net, "w_centers"):
        del style_net.w_centers
    style_net.register_buffer("w_centers", centers)
    ckpt_path = get_ckpt_paths(cfg.checkpoint_dir)[-1]
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt["EMA_generator"] = G.state_dict()
    torch.save(ckpt, ckpt_path)

compute_cluster_means()
    