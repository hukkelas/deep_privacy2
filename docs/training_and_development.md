# Training, development and Dataset Setup
## Training
First, download dataset for training (see below).

To start training, type the following:
```
python3 train.py configs/fdh/styleganL.py
```
The training automatically logs to [wandb](https://wandb.ai/).

## Model development utility scripts
**Dataset inspection:** To inspect the training dataset, you can use:
```
python3 -m tools.inspect_dataset configs/fdh/styleganL.py
```

**Sanity check:** 
```
python3 -m tools.dryrun configs/fdh/styleganL.py
```

**Output visualization:** To visualize output of trained models:
```
python3 -m tools.show_examples configs/fdh/styleganL.py
```


## Calculating metrics
```
python3 validate.py configs/fdh/styleganL.py
```
**NOTE:** The metrics calculated with validate.py will slightly differ from training metrics, as validate.py disables automatic mixed precision.


## Dataset Setup

**Setting Data directory:** 
The default dataset directory is ./data. If you want to change the dataset directory, set the environment variable `BASE_DATASET_DIR`. For example, `export BASE_DATASET_DIR=/work/data/`.


### FDF256
Follow the instructions [here](https://github.com/hukkelas/FDF/blob/master/FDF256.md) to download the FDF256 dataset. The dataset should be placed in the directory: `data/fdf256`.

### FDH
Follow the instructions [here](https://www.github.com/hukkelas/FDH) to download the FDH dataset. The dataset should be placed in the directory: `data/fdh`.

#### FDH for TriA-GAN paper

TriA-GAN requires the FDH dataset to be prepared for progressive growing training. This can be done by first downloading the dataset to 'data/fdh'.
Then, run:
```
python3 -m tools.dataset.copy_fdh_remove_embedding
```
Followed by
```
python3 -m tools.dataset.create_webdataset_progressive
```