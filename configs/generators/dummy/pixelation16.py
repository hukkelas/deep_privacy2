from dp2.generator.dummy_generators import PixelationGenerator
from tops.config import LazyCall as L

generator = L(PixelationGenerator)(pixelation_size=16)