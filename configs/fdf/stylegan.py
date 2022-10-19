from ..generators.stylegan_unet import generator
from ..datasets.fdf256 import data
from ..discriminators.sg2_discriminator import discriminator, G_optim, D_optim, loss_fnc
from ..defaults import train, common, EMA

train.max_images_to_train = int(35e6)
G_optim.lr = 0.002
D_optim.lr = 0.002
generator.input_cse = False
loss_fnc.r1_opts.lambd = 1
train.ims_per_val = int(2e6)

common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/89660f04-5c11-4dbf-adac-cbe2f11b0aeea25cbf78-7558-475a-b3c7-03f5c10b7934646b0720-ca0a-4d53-aded-daddbfa45c9e"
common.model_md5sum = "e8e32190528af2ed75f0cb792b7f2b07"