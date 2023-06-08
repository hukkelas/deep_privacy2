from dp2.generator.dummy_generators import GaussianBlurGenerator
from tops.config import LazyCall as L
from ...defaults import common

generator = L(GaussianBlurGenerator)()