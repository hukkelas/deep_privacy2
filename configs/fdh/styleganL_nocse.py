from tops.config import LazyCall as L
from ..generators.stylegan_unet import generator
from ..datasets.fdh import data
from ..discriminators.sg2_discriminator import discriminator, G_optim, D_optim, loss_fnc
from ..defaults import train, common, EMA

train.max_images_to_train = int(50e6)
G_optim.lr = 0.002
D_optim.lr = 0.002
generator.input_cse = False
data.load_embeddings = False
common.model_url = "https://folk.ntnu.no/haakohu/checkpoints/deep_privacy2/fdh_styleganL_nocse.ckpt"
common.model_md5sum = "fda0d809741bc67487abada793975c37"
generator.fix_errors = False