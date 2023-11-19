import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import tops


class ProjectedViT(nn.Module):

    def __init__(self, model_name, weight_path) -> None:
        super().__init__()
        model = timm.create_model(model_name, pretrained=weight_path is None)
        if weight_path is not None:
            filepath = tops.download_file(weight_path)
            state_dict = torch.load(filepath, map_location="cpu")
            state = state_dict["model"] if "model" in state_dict else state_dict
            model.load_state_dict(state, strict=False)
        start_index = 2 if "deit" in model_name else 1
        patch_size = [16, 16]
        if "tiny" in model_name:
            proj_features = [24, 48, 96, 192]
            layer_hooks = [2, 5, 8, 11]
            vit_dim = 192

        elif "small" in model_name:
            proj_features = [48, 96, 192, 384]
            layer_hooks = [2, 5, 8, 11]
            vit_dim = 384

        elif "base" in model_name:
            proj_features = [96, 192, 384, 768]
            layer_hooks = [2, 5, 8, 11]
            vit_dim = 768
        elif "large" in model_name:  # Depth 24
            proj_features = [256, 512, 1024, 1024]
            layer_hooks = [5, 11, 17, 23]
            vit_dim = 1024
        elif "huge" in model_name:  # Depth 32
            proj_features = [320, 640, 1280, 1280]
            layer_hooks = [8, 15, 23, 31]
            vit_dim = 1280
            patch_size = [14, 14]

        model.slice_start_idx = start_index  # Accessed in forward_flex
        model.patch_size = patch_size
        # We inject this function into the VisionTransformer instances so that
        # we can use it with interpolated position embeddings without modifying the library source.
        model.layer_hooks = layer_hooks
        self.model = model
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
        self.model.blocks = self.model.blocks[:max(self.model.layer_hooks)+1]

    def forward_vit(self, x):
        b, c, h, w = x.shape
        pos_embed = _resize_pos_embed(
            self.model.pos_embed, h // self.model.patch_size[1], w // self.model.patch_size[0], self.model.slice_start_idx
        )

        if hasattr(self.model.patch_embed, "backbone"):
            x = self.model.patch_embed.backbone(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.model.patch_embed.proj(x).flatten(2).transpose(1, 2)

        if hasattr(self.model, "dist_token") and self.model.dist_token is not None:
            cls_tokens = self.model.cls_token.expand(
                b, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            dist_token = self.model.dist_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.model.cls_token.expand(
                b, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.model.pos_drop(x)
        feats = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.model.layer_hooks:
                feats.append(x)
        return feats

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        features = self.forward_vit(x)
        features = [x[:, self.model.slice_start_idx:] for x in features]
        features = [  # n l c -> n c h w
            torch.transpose(x, 1, 2).unflatten(dim=-1, sizes=(h//self.model.patch_size[0], w//self.model.patch_size[1]))
            for x in features
        ]

        x0, x1, x2, x3 = features
        x0 = self.layer1(x0)
        x1 = self.layer2(x1)
        x2 = self.layer3(x2)
        x3 = self.layer4(x3)
        return x0, x1, x2, x3


def _make_vit_timm(model_name, weight_path=None):
    return ProjectedViT(model_name, weight_path)


def _resize_pos_embed(posemb: torch.Tensor, gs_h: int, gs_w: int, slice_start_idx: int) -> torch.Tensor:
    posemb_tok, posemb_grid = (
        posemb[:, : slice_start_idx],
        posemb[0, slice_start_idx:],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb
