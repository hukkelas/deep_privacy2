from tops.config import LazyCall as L 
from dp2.generator.deep_privacy1 import MSGGenerator
from ..datasets.fdf128 import data
from ..defaults import common, train

generator = L(MSGGenerator)()

common.model_url = "https://folk.ntnu.no/haakohu/checkpoints/fdf128_model512.ckpt"
common.model_md5sum = "6cc8b285bdc1fcdfc64f5db7c521d0a6"