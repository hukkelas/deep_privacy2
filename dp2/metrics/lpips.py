import torch
import tops
import sys
from contextlib import redirect_stdout
from torch_fidelity.sample_similarity_lpips import NetLinLayer, URL_VGG16_LPIPS, VGG16features, normalize_tensor, spatial_average


class SampleSimilarityLPIPS(torch.nn.Module):
    SUPPORTED_DTYPES = {
        'uint8': torch.uint8,
        'float32': torch.float32,
    }

    def __init__(self):

        super().__init__()
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        with redirect_stdout(sys.stderr):
            fp = tops.download_file(URL_VGG16_LPIPS)
            state_dict = torch.load(fp, map_location="cpu")
        self.load_state_dict(state_dict)
        self.net = VGG16features()
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        mean_rescaled = (1 + torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1)) * 255 / 2
        inv_std_rescaled = 2 / (torch.tensor([.458, .448, .450]).view(1, 3, 1, 1) * 255)
        self.register_buffer("mean", mean_rescaled)
        self.register_buffer("std", inv_std_rescaled)

    def normalize(self, x):
        # torchvision values in range [0,1] mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        x = (x.float() - self.mean) * self.std
        return x

    @staticmethod
    def resize(x, size):
        if x.shape[-1] > size and x.shape[-2] > size:
            x = torch.nn.functional.interpolate(x, (size, size), mode='area')
        else:
            x = torch.nn.functional.interpolate(x, (size, size), mode='bilinear', align_corners=False)
        return x

    def lpips_from_feats(self, feats0, feats1):
        diffs = {}
        for kk in range(self.L):
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk].model(diffs[kk])) for kk in range(self.L)]
        val = sum(res)
        return val

    def get_feats(self, x):
        assert x.dim() == 4 and x.shape[1] == 3, 'Input 0 is not Bx3xHxW'
        if x.shape[-2] < 16 or x.shape[-1] < 16:  # Resize images < 16x16
            f = 2
            size = tuple([int(f*_) for _ in x.shape[-2:]])
            x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        in0_input = self.normalize(x)
        outs0 = self.net.forward(in0_input)

        feats = {}
        for kk in range(self.L):
            feats[kk] = normalize_tensor(outs0[kk])
        return feats

    def forward(self, in0, in1):
        feats0 = self.get_feats(in0)
        feats1 = self.get_feats(in1)
        return self.lpips_from_feats(feats0, feats1), feats0, feats1
