import torch
import copy
import tops
from tops import logger
from .torch_utils import set_requires_grad


class EMA:
    """
    Expoenential moving average.
    See:
        Yazici, Y. et al.The unusual effectiveness of averaging in GAN training. ICLR 2019

    """

    def __init__(
            self,
            generator: torch.nn.Module,
            batch_size: int,
            rampup: float,
    ):
        self.rampup = rampup
        self._nimg_half_time = batch_size * 10 / 32 * 1000
        self._batch_size = batch_size
        with torch.no_grad():
            self.generator = copy.deepcopy(generator.cpu()).eval()
            self.generator = tops.to_cuda(self.generator)
        self.old_ra_beta = 0
        set_requires_grad(self.generator, False)

    def update_beta(self):
        y = self._nimg_half_time
        global_step = logger.global_step()
        if self.rampup != None:
            y = min(y, global_step*self.rampup)
        self.ra_beta = 0.5 ** (self._batch_size/max(y, 1e-8))
        if self.ra_beta != self.old_ra_beta:
            logger.add_scalar("stats/EMA_beta", self.ra_beta)
        self.old_ra_beta = self.ra_beta

    @torch.no_grad()
    def update(self, normal_G):
        with torch.autograd.profiler.record_function("EMA_update"):
            for ema_p, p in zip(self.generator.parameters(),
                                normal_G.parameters()):
                ema_p.copy_(p.lerp(ema_p, self.ra_beta))
            for ema_buf, buff in zip(self.generator.buffers(),
                                     normal_G.buffers()):
                ema_buf.copy_(buff)

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def __getattr__(self, name: str):
        if hasattr(self.generator, name):
            return getattr(self.generator, name)
        raise AttributeError(f"Generator object has no attribute {name}")

    def cuda(self, *args, **kwargs):
        self.generator = self.generator.cuda()
        return self

    def state_dict(self, *args, **kwargs):
        return self.generator.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.generator.load_state_dict(*args, **kwargs)

    def eval(self):
        self.generator.eval()

    def train(self):
        self.generator.train()

    @property
    def module(self):
        return self.generator.module

    def sample(self, *args, **kwargs):
        return self.generator.sample(*args, **kwargs)
