import torch
import torchvision
from setuptools import setup, find_packages

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 9], "Requires PyTorch >= 1.9"
torchvision_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert torchvision_ver >= [0, 11], "Requires torchvision >= 0.11"

setup(
    name="dp2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "cython",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "opencv-python",
        "detectron2-densepose@git+https://github.com/facebookresearch/detectron2@96c752ce821a3340e27edd51c28a00665dd32a30#subdirectory=projects/DensePose",
        "torch_fidelity==0.3.0",
        "ninja==1.10.2",
        "moviepy",
        "pyspng",
        "face_detection@git+https://github.com/hukkelas/DSFD-Pytorch-Inference",
        "wandb",
        "termcolor",
        "tops@git+https://github.com/hukkelas/torch_ops.git",
        "motpy@git+https://github.com/wmuron/motpy@c77f85d27e371c0a298e9a88ca99292d9b9cbe6b",
        "fast_pytorch_kmeans",
        "einops",
        "einops_exts",
        "regex",
        "setuptools==59.5.0",
        "resize_right==0.0.2",
        "pillow==8.3.1",
        "scipy==1.7.1",
        "webdataset==0.2.26",
        "scikit-image",
        "imageio==2.4.1",
        "timm==0.6.7",
        "clip@git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620",

    ],
)
