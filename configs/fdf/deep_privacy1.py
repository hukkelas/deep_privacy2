from tops.config import LazyCall as L 
from dp2.generator.deep_privacy1 import MSGGenerator
from ..datasets.fdf128 import data
from ..defaults import common, train

generator = L(MSGGenerator)()

common.model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/0940803d-1a2c-4b54-9c1d-0d3aeb80c5a8970f7dd4-bde6-4120-9d3e-5fad937eef2b7b0544e2-8c00-43c2-9193-f56679740872"
common.model_md5sum = "6cc8b285bdc1fcdfc64f5db7c521d0a6"