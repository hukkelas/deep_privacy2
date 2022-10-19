from tops.config import LazyCall as L
from ..generators.stylegan_unet import generator
from ..datasets.fdh import data
from ..discriminators.sg2_discriminator import discriminator, G_optim, D_optim, loss_fnc
from ..defaults import train, common, EMA

train.max_images_to_train = int(50e6)
train.batch_size = 64
G_optim.lr = 0.002
D_optim.lr = 0.002
data.train.loader.num_workers = 4
train.ims_per_val = int(1e6)
loss_fnc.r1_opts.lambd = .1

common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/21841da7-2546-4ce3-8460-909b3a63c58b13aac1a1-c778-4c8d-9b69-3e5ed2cde9de1524e76e-7aa6-4dd8-b643-52abc9f0792c"
common.model_md5sum = "3411478b5ec600a4219cccf4499732bd"