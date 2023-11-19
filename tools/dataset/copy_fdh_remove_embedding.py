from pathlib import Path
import webdataset as wds
import numpy as np
import tqdm
import io
from PIL import Image # This fixes a bug in webdataset writer...


def png_decoder(x):
    with Image.open(io.BytesIO(x)) as im:
        return np.array(im)


def kp_decoder(x):
    # Keypoints are between [0, 1] for webdataset
    keypoints = np.load(io.BytesIO(x))
    return keypoints


def process_dataset(source_path: Path, target_path: Path):
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
    target_path.mkdir(exist_ok=True, parents=True)

    writer = wds.ShardWriter(str(target_path.joinpath("out-%06d.tar")), maxsize=200*1024*1024, maxcount=3000) # maxsize 200MB

    for sample in tqdm.tqdm(pipeline):
        keypoints = sample["keypoints.npy"].astype(np.float32)
        im = sample["image.png"]
        keypoints[:, 0] /= im.shape[1]
        keypoints[:, 1] /= im.shape[0]
        writer.write({
            "__key__": sample["__key__"],
            "image.png": sample["image.png"],
            "mask.png": sample["mask.png"],
            "keypoints.npy": keypoints,
            "maskrcnn_mask.png": sample["maskrcnn_mask.png"],
        })
    writer.close()


def main():
    import os
    dataset_base_dir = (
        os.environ["BASE_DATASET_DIR"] if "BASE_DATASET_DIR" in os.environ else "data"
    )
    data_dir = Path(dataset_base_dir, "fdh")
    sources = [
        data_dir.joinpath("train", "out-{000000..001421}.tar"),
        data_dir.joinpath("val", "out-{000000..000023}.tar")
    ]
    target_path = [
        data_dir.parent.joinpath("fdh_no_embeddings", "train"),
        data_dir.parent.joinpath("fdh_no_embeddings", "val")
    ]
    for s, t in zip(sources, target_path):
        process_dataset(s, t)
    

if __name__ == "__main__":
    main()
