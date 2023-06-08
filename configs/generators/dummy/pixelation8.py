from dp2.generator.dummy_generators import PixelationGenerator
from tops.config import LazyCall as L
from ...defaults import common

generator = L(PixelationGenerator)(pixelation_size=8)