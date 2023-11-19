from pathlib import Path
import webdataset as wds
import numpy as np
import torch
import tqdm
import io
from PIL import Image # This fixes a bug in webdataset writer...
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode


def png_decoder(x):
    with Image.open(io.BytesIO(x)) as im:
        return np.array(im)


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = np.load(io.BytesIO(x))
    return keypoints


def process_dataset(source_path: Path, target_path: Path, imsizes):
    decoder = [
        wds.handle_extension("image.png", png_decoder),
        wds.handle_extension("mask.png", png_decoder),
        wds.handle_extension("maskrcnn_mask.png", png_decoder),
        wds.handle_extension("keypoints.npy", kp_decoder),
    ]
    pipeline = [
        wds.SimpleShardList(str(source_path)),
        wds.tarfile_to_samples(),
        wds.decode(*decoder),
    ]
    pipeline = wds.DataPipeline(*pipeline)

    # max size is number of bytes
    target_paths = [
        target_path.joinpath(str(s[0]))
        for s in imsizes
    ]
    for t in target_paths:
        t.mkdir(exist_ok=True, parents=True)

    writers = [
        wds.ShardWriter(str(t.joinpath("out-%06d.tar")), maxsize=200*1024*1024, maxcount=3000) # maxsize 200MB
        for t in target_paths
    ]

    for sample in tqdm.tqdm(pipeline, total=int(2e6)):
        keypoints = sample["keypoints.npy"].astype(np.float32)
        im = sample["image.png"]
        mask_float = sample["mask.png"][:, :, None].astype(np.uint8)
        mask = torch.from_numpy(sample["mask.png"])[None, None] > 0 
        assert mask.dtype == torch.bool
        maskrcnn_mask = torch.from_numpy(sample["maskrcnn_mask.png"])[None, None]

        condition = im * mask_float + ((1 - mask_float) * 127)
        assert condition.dtype == np.uint8
        condition = Image.fromarray(condition)
        im = Image.fromarray(im)
        for widx, imsize in enumerate(imsizes):
            im_ = np.array(im.resize(list(imsize)[::-1], resample=Image.BILINEAR))
            condition_ = np.array(condition.resize(list(imsize)[::-1], resample=Image.BILINEAR))
            mask_ = (torch.nn.functional.adaptive_max_pool2d(mask.logical_not().float(), output_size=imsize) > 0).logical_not()
            maskrcnn_mask_ = resize(maskrcnn_mask, imsize, InterpolationMode.NEAREST)
            assert mask_.dtype == torch.bool
            assert maskrcnn_mask.dtype == torch.bool
            to_replace = mask_.squeeze()[:, :, None].repeat(1, 1, 3).cpu().numpy()
            condition_[to_replace] = np.array(im_)[to_replace]
            writers[widx].write({
                "__key__": sample["__key__"],
                "image.png": im_,
                "mask.png": np.array(mask_.squeeze()),
                "keypoints.npy": keypoints,
                "maskrcnn_mask.png": np.array(maskrcnn_mask_).squeeze(),
                "condition.png": condition_
            })
    for w in writers:
        w.close()

def main():
    import os
    dataset_base_dir = (
        os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
    )
    data_dir = Path(dataset_base_dir, "fdh_no_embeddings")
    sources = [
        data_dir.joinpath("val", "out-{000000..000010}.tar"),
        data_dir.joinpath("train", "out-{000000..000612}.tar"),
    ]
    target_path = [
        data_dir.parent.joinpath("fdh_no_embeddings_resampled", "val"),
        data_dir.parent.joinpath("fdh_no_embeddings_resampled", "train"),
    ]
    imsizes = [
        (18, 10),
        (36, 20),
        (72, 40),
        (144, 80),
    ]
    for s, t in zip(sources, target_path):
        process_dataset(s, t, imsizes)
    

if __name__ == "__main__":
    main()