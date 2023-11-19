# Code adapted from: https://github.com/autonomousvision/projected_gan
from typing import List, Optional
import torch
import torch.nn as nn
from tops.config import instantiate


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, expand):
        super().__init__()

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor]):
        output = x0
        if x1 is not None:
            output = self.skip_add.add(output, x1)
        output = nn.functional.interpolate(
            output, scale_factor=2., mode="bilinear", align_corners=True)
        return self.out_conv(output)


class ProjectorCSM(nn.Module):
    def __init__(self, pretrained, cout, proj_type: int) -> None:
        super().__init__()
        assert proj_type in [0, 1, 2], "Invalid projection type"
        im_res = 256
        tmp = torch.zeros(1, 3, im_res, im_res)
        self.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]
        in_channels = [out.shape[1] for out in pretrained(tmp)]
        self.CHANNELS = in_channels
        if proj_type == 0:
            return
        out_channels = [cout, cout*2, cout*4, cout*8]
        self.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1)
        self.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1)
        self.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1)
        self.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1)
        self.CHANNELS = out_channels
        # Build CCM
        if proj_type == 1:
            return
        # build CSM
        self.layer3_csm = FeatureFusionBlock(out_channels[3], expand=True)
        self.layer2_csm = FeatureFusionBlock(out_channels[2], expand=True)
        self.layer1_csm = FeatureFusionBlock(out_channels[1],  expand=True)
        self.layer0_csm = FeatureFusionBlock(out_channels[0], expand=False)
        self.CHANNELS = [cout, cout, cout*2, cout*4]
        # CSM upsamples x2 so the feature map resolution doubles
        self.RESOLUTIONS = [res*2 for res in self.RESOLUTIONS]
        self.proj_type = proj_type

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x0, x1, x2, x3 = x
        if self.proj_type == 0:
            return x
        out0_channel_mixed = self.layer0_ccm(x0)
        out1_channel_mixed = self.layer1_ccm(x1)
        out2_channel_mixed = self.layer2_ccm(x2)
        out3_channel_mixed = self.layer3_ccm(x3)
        if self.proj_type == 1:
            return [out0_channel_mixed, out1_channel_mixed, out2_channel_mixed, out3_channel_mixed]
        # from bottom to top
        out3_scale_mixed = self.layer3_csm(out3_channel_mixed, None)
        out2_scale_mixed = self.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.layer0_csm(out1_scale_mixed, out0_channel_mixed)
        return [out0_scale_mixed, out1_scale_mixed, out2_scale_mixed, out3_scale_mixed]


class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        interp_size,
        input_BGR,
        jit_script,
        mean=None,
        std=None,
        cout=64,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        eval_mode=True
    ):
        super().__init__()
        self.pretrained = instantiate(backbone_cfg)
        if eval_mode:
            self.pretrained = self.pretrained.eval()
        if mean is None:
            mean = self.pretrained.mean
            std = self.pretrained.std
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        # build pretrained feature network and random decoder (scratch)
        self.projector = ProjectorCSM(self.pretrained, cout, proj_type)
        self.CHANNELS = self.projector.CHANNELS
        self.RESOLUTIONS = self.projector.RESOLUTIONS
        self.input_BGR = input_BGR
        self.interp_size = interp_size
        self.jit_script = jit_script
        if interp_size is not None:
            self.interp_size = list(interp_size)

    def forward(self, x):
        if self.interp_size is not None:
            x = nn.functional.interpolate(x, self.interp_size, mode="bilinear", align_corners=True)
        if self.input_BGR:
            x = x.flip(1)
        features = self.pretrained(x)
        return self.projector(features)
