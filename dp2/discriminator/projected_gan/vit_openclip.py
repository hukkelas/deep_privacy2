import torch
import torch.nn as nn
try:
    import open_clip
    from open_clip.model import VisualTransformer
except ImportError:
    print("Could not import openclip")
from einops import rearrange
from .vit import _resize_pos_embed


class OpenCLIPViT(nn.Module):

    def __init__(self, model_name, pretrained_dataset) -> None:
        super().__init__()
        assert model_name in [
            "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-g-14"
        ]
        assert model_name not in ["ViT-H-14", "ViT-g-14"]
        if model_name == "ViT-B-16":
            proj_features = [96, 192, 384, 768]
            layer_hooks = [2, 5, 8, 11]
            vit_dim = 768
            patch_size = [16, 16]
        if model_name == "ViT-L-14":
            patch_size = [14, 14]
            vit_dim = 1024
            layer_hooks = [5, 11, 17, 23]
            proj_features = [256, 512, 1024, 1024]
        if model_name == "ViT-B-32":
            patch_size = [32, 32]
            vit_dim = 768
            proj_features = [96, 192, 384, 768]
            layer_hooks = [2, 5, 8, 11]
        # preprocess consists of resize -> center crop -> rgb -> normalize
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained_dataset)
        model = model.visual
        print(model)

        self.slice_start_idx = 1  # Accessed in forward_flex
        self.patch_size = patch_size
        # We inject this function into the VisionTransformer instances so that
        # we can use it with interpolated position embeddings without modifying the library source.
        model.layer_hooks = layer_hooks
        self.model = model
        self.model.transformer.resblocks = self.model.transformer.resblocks[:max(self.model.layer_hooks)+1]
        if patch_size[0] == 32:
            self._make_proj_layers_vit32(vit_dim, proj_features)
        else:
            self._make_proj_layers(vit_dim, proj_features)

    def _make_proj_layers_vit32(self, vit_dim, proj_features):
        self.layer1 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[0], 1),
            nn.ConvTranspose2d(proj_features[0], proj_features[0], 8, stride=8)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[1], 1),
            nn.ConvTranspose2d(proj_features[1], proj_features[1], 4, stride=4),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[2], 1),
            nn.ConvTranspose2d(proj_features[2], proj_features[2], 2, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[3], 1),
        )

    def _make_proj_layers(self, vit_dim, proj_features):
        self.layer1 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[0], 1),
            nn.ConvTranspose2d(proj_features[0], proj_features[0], 4, stride=4)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[1], 1),
            nn.ConvTranspose2d(proj_features[1], proj_features[1], 2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[2], 1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(vit_dim, proj_features[3], 1),
            nn.Conv2d(proj_features[3], proj_features[3], 3, stride=2, padding=1),
        )

    def forward_vit(self, x):
        assert isinstance(self.model, VisualTransformer)
        b, c, h, w = x.shape
        pos_embed = _resize_pos_embed(
            self.model.positional_embedding[None], h // self.patch_size[1], w // self.patch_size[0], self.slice_start_idx
        )[0]

        x = self.model.conv1(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        x = torch.cat(
            [self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + pos_embed
        x = self.model.ln_pre(x)
        x = rearrange(x, "b l c -> l b c")
        feats = []
        for i, l in enumerate(self.model.transformer.resblocks):
            x = l(x)
            if i in self.model.layer_hooks:
                feats.append(x)
        return feats

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        features = self.forward_vit(x)  # Outputs l n  c
        features = [x[self.slice_start_idx:] for x in features]
        features = [
            rearrange(f, "(h w) n c -> n c h w", h=h//self.patch_size[0])
            for f in features
        ]

        x0, x1, x2, x3 = features
        x0 = self.layer1(x0)
        x1 = self.layer2(x1)
        x2 = self.layer3(x2)
        x3 = self.layer4(x3)
        return x0, x1, x2, x3
