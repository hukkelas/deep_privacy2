# Getting Started

## Requirements
- Linux and MacOS (CPU only) are supported, but we recommend linux. The code is not tested on Windows, and it might be difficult to install detectron2 for windows (see [here](https://github.com/hukkelas/deep_privacy2/issues/10) for issue).
- GPU is not required for inference.
- 1-8 server-grade NVIDIA GPUs with at least 32GB of memory for training.
- Python >= 3.8.

## Install with Anaconda
We recommend to setup and install pytorch with [anaconda](https://www.anaconda.com/) following the [pytorch installation instructions](https://pytorch.org/get-started/locally/).

1. Clone repository: `git clone https://github.com/hukkelas/deep_privacy2/`.
2. Install using `setup.py`:
```
pip install -e .
```
or:
```
pip install git+https://github.com/hukkelas/deep_privacy2/
```

##  Install with Docker

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to support GPU acceleration.
2. Build the docker image using the [Dockerfile](Dockerfile).
```bash
# If you're not planning to train the network (or not use wandb logging), you can remove the WANDB_API_KEY argument.
docker build -t deep_privacy2 --build-arg WANDB_API_KEY=YOUR_WANDB_KEY  --build-arg UID=$(id -u) --build-arg UNAME=$(id -un) .
```
3. Run the docker image with selected command:
```
docker run --runtime=nvidia --gpus '"device=0"' --name deep_privacy2 --ipc=host -u $(id -u) -v $PWD:/home/$(id -un) --rm deep_privacy2 python3 anonymize.py configs/anonymizers/deep_privacy1.py -i media/regjeringen.jpg -o output.png
```


## Environment variables
DeepPrivacy2 uses the following environment variables:

- `BASE_DATASET_DIR`: The directory where datasets are located. Defaults to `./data` if not set
- `BASE_OUTPUT_DIR`: The directory where outputs will be saved. Defaults to `./outputs`.
- `FBA_METRICS_CACHE`: The directory where intermediate calculations from metric calculation is saved. For example, FID statistics for real images.
- `TORCH_HOME`: The directory where checkpoints downloaded from URL are saved.
