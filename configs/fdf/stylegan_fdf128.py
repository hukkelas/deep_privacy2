from ..discriminators.sg2_discriminator import discriminator, G_optim, D_optim, loss_fnc
from ..datasets.fdf128 import data
from ..generators.stylegan_unet import generator
from ..defaults import train, common, EMA
from tops.config import LazyCall as L

G_optim.lr = 0.002
D_optim.lr = 0.002
generator.update(cnum=128, max_cnum_mul=4, input_cse=False)
loss_fnc.r1_opts.lambd = 0.1

train.update(ims_per_val=int(2e6), batch_size=64, max_images_to_train=int(35e6))

common.update(
    model_url="https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/66d803c0-55ce-44c0-9d53-815c2c0e6ba4eb458409-9e91-45d1-bce0-95c8a47a57218b102fdf-bea3-44dc-aac4-0fb1d370ef1c",
    model_md5sum="bccd4403e7c9bca682566ff3319e8176"
)