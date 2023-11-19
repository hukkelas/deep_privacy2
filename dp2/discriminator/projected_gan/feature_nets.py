# Code adapted from: https://github.com/autonomousvision/projected_gan
from typing import List
import torchvision
import torch
import torch.nn as nn
import timm
import tops
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_names) -> None:
        super().__init__()
        self.model = create_feature_extractor(model, return_nodes=layer_names)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.model(x).values())


class FeatureExtractorByLayers(nn.Module):

    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x) -> List[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


def _make_efficientnet(model_type):
    model = timm.create_model(model_type, pretrained=True)
    layer0 = nn.Sequential(
        model.conv_stem, model.bn1, *model.blocks[0:2]
    )
    layer1 = nn.Sequential(*model.blocks[2:3])
    layer2 = nn.Sequential(*model.blocks[3:5])
    layer3 = nn.Sequential(*model.blocks[5:9])
    return FeatureExtractorByLayers(layer0, layer1, layer2, layer3)


def _make_resnet50(weight_path=None):
    model = torchvision.models.resnet50(pretrained=weight_path is None)
    if weight_path is not None:
        state_dict = tops.load_file_or_url(weight_path)
        model.load_state_dict(state_dict)
    return FeatureExtractor(
        model, ["layer1", "layer2", "layer3", "layer4"]
    )


def _make_resnet50_swav(weight_path):
    model = torch.hub.load("facebookresearch/swav", weight_path)
    return FeatureExtractor(
        model, ["layer1", "layer2", "layer3", "layer4"]
    )


def _make_resnet50_clip(clip_type="RN50"):
    import clip
    model, preprocess = clip.load(clip_type, device="cpu")
    return FeatureExtractor(
        model.visual,
        layer_names=["layer1", "layer2", "layer3", "layer4"]
    )


def _make_resnet50_cse(cfg_url, **kwargs):
    from dp2.detection.models.cse import CSEDetector
    model = CSEDetector(cfg_url=cfg_url).model.cpu()
    return FeatureExtractorByLayers(
        nn.Sequential(
            model.backbone.bottom_up.stem,
            model.backbone.bottom_up.res2,
        ),
        model.backbone.bottom_up.res3,
        model.backbone.bottom_up.res4,
        model.backbone.bottom_up.res5
    )
