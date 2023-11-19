# Anonymizing datasets

This document contains information to reproduce the experiments from the paper "Does Image Anonymization Impact Computer Vision Training?" published at CVPR Workshop on Autonomous Driving 2023.

## Dataset setup
The paper evaluates on BDD100k, Cityscapes, and COCO. Download the datasets following the instructions from each, and keep the original datafile structure as documented for each dataset. The code expects the following folder structre for the different datasets:

### BDD100k
```
bdd100k-directory/
    annotated_keypoints_train.json # Can be downloaded from https://drive.google.com/file/d/1apTK26BuERnruEDqWJ2LdLQ9xoM-aRRw/view?usp=sharing
    face_boxes_train.json # Can be downloaded from https://drive.google.com/file/d/1D2p1g8mC-0qu9AqHCIsTmvqcxBz9Ug-t/view?usp=sharing
    images/
    jsons/
    labels/
```
### COCO
```
coco-directory/
    boxes_train2017.json # Can be downloaded from https://drive.google.com/file/d/1FS9ZOymm7XMmZd_ocH-ikYioJlH6J63c/view?usp=sharing
    boxes_val2017.json # Can be downloaded from https://drive.google.com/file/d/12B5a_apfUOr8UJK8ED2g1EX6kU1IxWsw/view?usp=sharing
    annotations/
        person_keypoints_train2017.json
        captions_val2017.json
        instances_val2017.json
        captions_train2017.json
        person_keypoints_val2017.json
        instances_train2017.json
    train2017/
        ...
    val2017/
        ...
        
```
### Cityscapes
```
cityscapes-directory/
    annotated_keypoints_train.json # Can be downloaded from https://drive.google.com/file/d/1Acv7A1hONHUH1MvaXwCHQ95-f5LUEl7c/view?usp=sharing
    annotated_keypoints_val.json # Can be downloaded from  https://drive.google.com/file/d/1WkfIhuKlcie9DjMt5IgL10FzIwUFrtkX/view?usp=sharing
    face_boxes_train.json # Can be downloaded from https://drive.google.com/file/d/1O56H4xbx_DgIkiF1E8JTZUjYmuHgO0Bs/view?usp=sharing
    camera/
    gtFine/
    leftImg8bit/
```
## Anonymizing
All scripts for full-body anonymization accepts the following three config files: [configs/anonymizers/FB_triagan.py](configs/anonymizers/FB_triagan.py), [configs/anonymizers/traditional/body/gaussian.py](configs/anonymizers/traditional/body/gaussian.py), and [configs/anonymizers/traditional/body/maskout.py](configs/anonymizers/traditional/body/maskout.py).
All scripts for face anonymization accepts the following three config files: [configs/anonymizers/face_fdf128.py](configs/anonymizers/face_fdf128.py), [configs/anonymizers/traditional/face/gaussian.py](configs/anonymizers/traditional/face/gaussian.py), and [configs/anonymizers/traditional/face/maskout.py](configs/anonymizers/traditional/face/maskout.py).
All scripts accepts the following config files: 

Some of the scripts have additional arguments:
- `--match-histogram`: Enables naive histogram matching
- `--sampler`: Select the sampling approach of the latent variable
- `--n-sampling-steps`: Setting this to > 1 enables histogram matching via latent optimization. Default is set to 100 for experiments in paper.
- `--min_size`: Set the minimum resolution of bboxes to anonymize.

**Note**: The path to the dataset and directory to save the anonymized dataset is hard-coded in the code. Change these yourself, all dataset sources and targets are saved under the path `/mnt/work2/haakohu/datasets` for my setup.

### Anonymizing COCO

**Face Anonymization:**

```
python3 -m tools.anonymization.coco.anonymize_faces configs/anonymizers/face_fdf128.py
```

We experimented with higher-resolution face anonymization for the COCO dataset. To refine the face anonymization, run:
```
python3 -m tools.anonymization.coco.refine_faces
```
Note that this has to be run **after**  the low-resolution face anonymizer.

**Full-body Anonymization:**
```
python3 -m tools.anonymization.coco.anonymize_body configs/anonymizers/FB_triagan.py
```

### Anonymizing BDD100k

**Face Anonymization:**

```
python3 -m tools.anonymization.bdd100k.anonymize_face configs/anonymizers/face_fdf128.py
```

**Full-body Anonymization:**
```
python3 -m tools.anonymization.bdd100k.anonymize_body configs/anonymizers/FB_triagan.py
```


### Anonymizing Cityscapes

**Face Anonymization:**

```
python3 -m tools.anonymization.cityscapes.anonymize_faces configs/anonymizers/face_fdf128.py
```

**Full-body Anonymization:**
```
python3 -m tools.anonymization.cityscapes.anonymize_body configs/anonymizers/FB_triagan.py
```


## Training the models on anonymized datasets

See the paper to reproduce our experiments to train and validate on the anonymized datasets.