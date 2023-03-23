import tops
from dp2 import utils
from pathlib import Path
from torch_fidelity.generative_model_modulewrapper import GenerativeModelModuleWrapper
import torch
import torch_fidelity


class GeneratorIteratorWrapper(GenerativeModelModuleWrapper):

    def __init__(self, generator, dataloader, zero_z: bool, n_diverse: int):
        if isinstance(generator, utils.EMA):
            generator = generator.generator
        z_size = generator.z_channels
        super().__init__(generator, z_size, "normal", 0)
        self.zero_z = zero_z
        self.dataloader = iter(dataloader)
        self.n_diverse = n_diverse
        self.cur_div_idx = 0

    @torch.no_grad()
    def forward(self, z, **kwargs):
        if self.cur_div_idx == 0:
            self.batch = next(self.dataloader)
        if self.zero_z:
            z = z.zero_()
        self.cur_div_idx += 1
        self.cur_div_idx = 0 if self.cur_div_idx == self.n_diverse else self.cur_div_idx
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            img = self.module(**self.batch)["img"]
            img = (utils.denormalize_img(img)*255).byte()
            return img


def compute_fid(generator, dataloader, real_directory, n_source, zero_z, n_diverse):
    generator = GeneratorIteratorWrapper(generator, dataloader, zero_z, n_diverse)
    batch_size = dataloader.batch_size
    num_samples = (n_source * n_diverse) // batch_size * batch_size
    assert n_diverse >= 1
    assert (not zero_z) or n_diverse == 1
    assert num_samples % batch_size == 0
    assert n_source <= batch_size * len(dataloader), (batch_size*len(dataloader), n_source, n_diverse)
    metrics = torch_fidelity.calculate_metrics(
        input1=generator,
        input2=real_directory,
        cuda=torch.cuda.is_available(),
        fid=True,
        input2_cache_name="_".join(Path(real_directory).parts) + "_cached",
        input1_model_num_samples=int(num_samples),
        batch_size=dataloader.batch_size
    )
    return metrics["frechet_inception_distance"]
