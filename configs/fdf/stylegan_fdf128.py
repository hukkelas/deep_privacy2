from ..discriminators.sg2_discriminator import discriminator, G_optim, D_optim, loss_fnc
from ..datasets.fdf128 import data
from ..generators.stylegan_unet import generator
from ..defaults import train, common, EMA
from tops.config import LazyCall as L

train.max_images_to_train = int(25e6)
G_optim.lr = 0.002
D_optim.lr = 0.002
generator.cnum = 128
generator.max_cnum_mul = 4
generator.input_cse = False
loss_fnc.r1_opts.lambd = .1
