from dp2.generator.dummy_generators import MaskOutGenerator
from tops.config import LazyCall as L
from ...defaults import common

generator = L(MaskOutGenerator)(noise="constant")