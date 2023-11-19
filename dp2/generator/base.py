import torch
import numpy as np
import tqdm
import tops
from ..layers import Module
from ..layers.sg2_layers import FullyConnectedLayer


class BaseGenerator(Module):

    def __init__(self, z_channels: int):
        super().__init__()
        self.z_channels = z_channels
        self.latent_space = "Z"

    @torch.no_grad()
    def get_z(
            self,
            x: torch.Tensor = None,
            z: torch.Tensor = None,
            truncation_value: float = None,
            batch_size: int = None,
            dtype=None, device=None) -> torch.Tensor:
        """Generates a latent variable for generator. 
        """
        if z is not None:
            return z
        if x is not None:
            batch_size = x.shape[0]
            dtype = x.dtype
            device = x.device
        if device is None:
            device = tops.get_device()
        if truncation_value == 0:
            return torch.zeros((batch_size, self.z_channels), device=device, dtype=dtype)
        z = torch.randn((batch_size, self.z_channels), device=device, dtype=dtype)
        if truncation_value is None:
            return z
        while z.abs().max() > truncation_value:
            m = z.abs() > truncation_value
            z[m] = torch.rand_like(z)[m]
        return z

    def sample(self, truncation_value, z=None, **kwargs):
        """
            Samples via interpolating to the mean (0).
        """
        if truncation_value is None:
            return self.forward(**kwargs)
        truncation_value = max(0, truncation_value)
        truncation_value = min(truncation_value, 1)
        if z is None:
            z = self.get_z(kwargs["condition"])
        z = z * truncation_value
        return self.forward(**kwargs, z=z)


class SG2StyleNet(torch.nn.Module):
    def __init__(self,
                 z_dim,                      # Input latent (Z) dimensionality.
                 w_dim,                      # Intermediate latent (W) dimensionality.
                 num_layers=2,        # Number of mapping layers.
                 lr_multiplier=0.01,     # Learning rate multiplier for the mapping layers.
                 w_avg_beta=0.998,    # Decay for tracking the moving average of W during training.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        # Construct layers.
        features = [self.z_dim] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, update_emas=False, **kwargs):
        tops.assert_shape(z, [None, self.z_dim])

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)
        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.float().detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d},  w_dim={self.w_dim:d}'

    def update_w(self, n=int(10e3), batch_size=32):
        """
            Calculate w_ema over n iterations.
            Useful in cases where w_ema is calculated incorrectly during training.
        """
        n = n // batch_size
        for i in tqdm.trange(n, desc="Updating w"):
            z = torch.randn((batch_size, self.z_dim), device=tops.get_device())
            self(z, update_emas=True)

    def get_truncated(self, truncation_value, condition=None, z=None, n=None,**kwargs):
        if n is None:
            n = condition.shape[0] if condition is not None else z.shape[0]
        if z is None:
            z = torch.randn((n, self.z_dim), device=tops.get_device())
        w = self(z)
        truncation_value = max(0, truncation_value)
        truncation_value = min(truncation_value, 1)
        return self.w_avg.to(w.dtype).lerp(w, truncation_value)

    def multi_modal_truncate(self, truncation_value, condition=None, w_indices=None, z=None, n=None,**kwargs):
        truncation_value = max(0, truncation_value)
        truncation_value = min(truncation_value, 1)
        if n is None:
            n = len(w_indices) if w_indices is not None else (condition.shape[0] if condition is not None else z.shape[0])
        if z is None:
            z = torch.randn((n, self.z_dim), device=tops.get_device())
        w = self(z)
        if w_indices is None:
            w_indices = np.random.randint(0, len(self.w_centers), size=(len(w)))
        w_centers = self.w_centers[w_indices].to(w.device)
        w = w_centers.to(w.dtype).lerp(w, truncation_value)
        return w

class BaseStyleGAN(BaseGenerator):

    def __init__(self, z_channels: int, w_dim: int):
        super().__init__(z_channels)
        self.style_net = SG2StyleNet(z_channels, w_dim)
        self.latent_space = "W"

    def get_w(self, z, update_emas):
        return self.style_net(z, update_emas=update_emas)

    @torch.no_grad()
    def sample(self, truncation_value, **kwargs):
        if truncation_value is None:
            return self.forward(**kwargs)
        w = self.style_net.get_truncated(truncation_value, **kwargs)
        return self.forward(**kwargs, w=w)

    def update_w(self, *args, **kwargs):
        self.style_net.update_w(*args, **kwargs)

    @torch.no_grad()
    def multi_modal_truncate(self, truncation_value, w_indices=None, **kwargs):
        w = self.style_net.multi_modal_truncate(truncation_value, w_indices=w_indices, **kwargs)
        return self.forward(**kwargs, w=w)
