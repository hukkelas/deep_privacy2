# Anonymization

## Available Models
DeepPrivacy2 provides various anonymization models. Each model has a training config and an anonymization config.
- Training Config details the training setup for the generative model.
- Anonymization config details the inference setup for anonymization. The anonymization config can link to several training configs when multi-modal anonymization is used. See [here](anonymization.md#multi-modal-anonymization) for more info about multi-modal anonymization.

### Face Anonymization Models
| Dataset (Resolution) | Detection Type          | Anonymization Config                 | Training Config                | Comment |
|----------------------|-------------------------|--------------------------------------|--------------------------------| |
|FDF128 (128x128)     | Face Bbox + 7 Keypoints | [configs/anonymizers/deep_privacy1.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/deep_privacy1.py) | *[configs/fdf/deep_privacy1.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdf/deep_privacy1.py)   | Not recommended unless you have large head rotations.|
| FDF128 (128x128)     | Face Bbox               | [configs/anonymizers/face_fdf128.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/face_fdf128.py)   | [configs/fdf/stylegan_fdf128.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdf/stylegan_fdf128.py) | Recommended model if all faces have a resolution < 128x128.|
| FDF256 (256x256)     | Face Bbox               | [configs/anonymizers/face.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/face.py)   | [configs/fdf/stylegan.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdf/stylegan.py) | Recommended model if any face has a resolution > 128x128.|

\*Note that the weights for [configs/fdf/deep_privacy1.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdf/deep_privacy1.py) are not available to retrain, as these weights are ported from the original [DeepPrivacy repository](s://github.com/hukkelas/DeepPrivacy).

The **face bbox** is the bounding box detected by [DSFD](http://github.com/hukkelas/DSFD-Pytorch-Inference).
The **7 keypoints** are the first 7 keypoints detected by a Keypoint R-CNN model trained on COCO, indicating nose, eyes, ears, and shoulders.

### Full-body Anonymization Models

| Detection Type | Anonymization Config | Training Config | Comment |
|---|---|---| |
| Segmentation Mask | [configs/anonymizers/FB_nocse.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/FB_nocse.py) | [configs/fdh/styleganL_nocse.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/configs/fdh/styleganL_nocse.py) | Generally produces poor quality human figures except for masks that explicitly encodes the pose of the person. |
| CSE + Segmentation Mask | [configs/anonymizers/FB_cse.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/FB_cse.py) | [configs/fdh/styleganL.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/configs/fdh/styleganL.py) | Recommended model. |

The FDH dataset with a resolution of 288x160 is used to train all full-body anonymization models.

The **CSE** refers to surface maps from a Continuous Surface Embedding model. **Segementation mask** refers to a instance segmentation mask, where we use the output of a Mask R-CNN model trained on COCO.
The **17 keypoints** are keypoints following the COCO format.

## Using anonymize.py
[anonymize.py](anonymize.py) is the main script for anonymization.

The typical usage is
```
python3 anonymize.py configs/anonymizers/FB_cse.py -i path_to_image.png
```
where the first argument is the chosen anonymizer (see above for the different models) and the second a path to an image/folder/video.

There are several optional arguments, see `python3 anonymize.py --help` for more info.
```
python3 anonymize.py --help
Usage: anonymize.py [OPTIONS] CONFIG_PATH

  config_path: Specify the path to the anonymization model to use.

Options:
  -i, --input_path TEXT           Input path. Accepted inputs: images, videos,
                                  directories.
  -o, --output_path PATH          Output path to save. Can be directory or
                                  file.
  -v, --visualize                 Visualize the result
  --max-res INTEGER               Maximum resolution  of height/wideo
  --start-time, --st INTEGER      Start time (second) for vide anonymization
  --end-time, --et INTEGER        End time (second) for vide anonymization
  --fps INTEGER                   FPS for anonymization
  --detection-score-threshold, --dst FLOAT RANGE
                                  Detection threshold, threshold applied for
                                  all detection models.  [0<=x<=1]
  --visualize-detection, --vd     Visualize only detections without running
                                  anonymization.
  --multi-modal-truncation, --mt  Enable multi-modal truncation proposed by:
                                  https://arxiv.org/pdf/2202.12211.pdf
  --cache                         Enable detection caching. Will save and load
                                  detections from cache.
  --amp                           Use automatic mixed precision for generator
                                  forward pass
  -t, --truncation_value FLOAT RANGE
                                  Latent interpolation truncation value.
                                  [0<=x<=1]
  --track                         Track detections over frames. Will use the
                                  same latent variable (z) for tracked
                                  identities.
  --seed INTEGER                  Set random seed for generating images.
  --person-generator PATH         Config path to unconditional person
                                  generator
  --cse-person-generator PATH     Config path to CSE-guided person generator
  --webcam                        Read image from webcam feed.
  --help                          Show this message and exit.
```

### Singe image anonymization
```
python3 anonymize.py configs/anonymizers/FB_cse.py -i path_to_image.png --output_path output.png
```

### Folder anonymization

If a folder is given as the input, all image and video files in the given folder will be anonymized and placed under --output_path. The script will duplicate the directory structure/filenames in the given folder for the output.
```
python3 anonymize.py configs/anonymizers/FB_cse.py -i path/to/input/folder --output_path output_folder
```

### Video anonymization
```
python3 anonymize.py configs/anonymizers/FB_cse.py -i path_to_video.mp4 --output_path output.mp4
```

### Webcam anonymization
```
python3 anonymize.py configs/anonymizers/FB_cse.py --webcam
```


## Multi-modal anonymization

DeepPrivacy2 supports anonymizing individuals detected by different modalities (e.g. faces and bodies).

![](media/header.png)

The following multi-modal anonymization models are provided: 


| Config File | Detection Modalities | Synthesis Configs | Modalities |
| --- |---| ---| --- |
| [configs/anonymizers/FB_cse_mask_face.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/FB_cse_mask_face.py) | CSE+Segmentation Masks, Segmentation Masks, Face bbox | [configs/fdh/styleganL.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdh/styleganL.py), [configs/fdh/styleganL_nocse.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdh/styleganL_nocse.py), [configs/fdf/stylegan.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdf/stylegan.py) |  Bodies (detected w/CSE), bodies (not detected w/CSE), Faces|
| [configs/anonymizers/FB_cse_mask.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/FB_cse_mask.py) | CSE+Segmentation Masks, Segmentation Masks | [configs/fdh/styleganL.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdh/styleganL.py), [configs/fdh/styleganL_nocse.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/fdh/styleganL_nocse.py), |  Bodies (detected w/CSE), bodies (not detected w/CSE)|

Note that combining different synthesis networks listed [previously](anonymization.md#available-models) is straightforward, see for eaxmple [configs/anonymizers/FB_cse_mask_face.py](https://github.com/hukkelas/deep_privacy2/blob/master/configs/anonymizers/FB_cse_mask_face.py).
