FROM nvcr.io/nvidia/pytorch:22.08-py3
ARG UID=1000
ARG UNAME=testuser
ARG WANDB_API_KEY
RUN useradd -ms /bin/bash -u $UID $UNAME && \
    mkdir -p /home/${UNAME} &&\
    chown -R $UID /home/${UNAME}
WORKDIR /home/${UNAME}
ENV DEBIAN_FRONTEND="noninteractive"
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV TORCH_HOME=/home/${UNAME}/.cache

# OPTIONAL - DeepPrivacy2 uses these environment variables to set directories outside the current working directory
#ENV BASE_DATASET_DIR=/work/haakohu/datasets
#ENV BASE_OUTPUT_DIR=/work/haakohu/outputs
#ENV FBA_METRICS_CACHE=/work/haakohu/metrics_cache

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  qt5-default -y
RUN pip install git+https://github.com/facebookresearch/detectron2@96c752ce821a3340e27edd51c28a00665dd32a30#subdirectory=projects/DensePose
COPY setup.py setup.py
RUN pip install \
    numpy>=1.20 \
    matplotlib \
    cython \
    tensorboard \
    tqdm \
    ninja==1.10.2 \
    opencv-python-headless==4.5.5.64 \
    ftfy==6.1.3 \
    moviepy \
    pyspng \
    git+https://github.com/hukkelas/DSFD-Pytorch-Inference \
    wandb \ 
    termcolor \
    git+https://github.com/hukkelas/torch_ops.git \
    git+https://github.com/wmuron/motpy@c77f85d27e371c0a298e9a88ca99292d9b9cbe6b \
    fast_pytorch_kmeans \
    einops_exts  \ 
    einops \ 
    regex \
    setuptools==59.5.0 \
    resize_right==0.0.2 \
    pillow \
    scipy==1.7.1 \
    webdataset==0.2.26 \
    scikit-image \
    timm==0.6.7
RUN pip install --no-deps torch_fidelity==0.3.0 clip@git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620
