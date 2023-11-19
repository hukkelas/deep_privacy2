# Reproducing results from: "Synthesizing Anyone, Anywhere, in Any Pose"

The dataset for FDH has to be set up for the triagan paper. See [training_and_development](here) for more info.

See the supplementary material for the paper for more information.

### Table 1 results:
#### Training
```
# Model A 
python3 train.py configs/fdh/triagan/ablations/fnets/im72_effnet.py
# B
python3 train.py configs/fdh/triagan/ablations/fnets/im72_effnet_MAD.py
# C
python3 train.py configs/fdh/triagan/ablations/fnets/im72_MAEL_clip_MAD.py
# D - requires progressive growing
python3 train.py configs/fdh/triagan/ablations/PG/im18.py
python3 train.py configs/fdh/triagan/ablations/PG/im36.py
python3 train.py configs/fdh/triagan/ablations/PG/im72.py
# E - required progrssive growing
python3 train.py configs/fdh/triagan/L_im18.py
python3 train.py configs/fdh/triagan/L_im36.py
python3 train.py configs/fdh/triagan/L_im72.py
python3 train.py configs/fdh/triagan/L_im144.py
python3 train.py configs/fdh/triagan/L_im288.py
```

#### Validation
For each config path, you can run:
```
python3 -m tools.triagan.validate_triaGAN
```
All table 1 results are computed from the model with resolution (72x40)

### Figure 3
All models in figure 3 is included in `configs/fdh/triagan/ablations/fnets`


### Table 2 results
```
python3 -m tools.triagan.validate_triaGAN configs/fdh/styleganL.py
python3 -m tools.triagan.validate_triaGAN configs/fdh/triagan/L_im288.py
```