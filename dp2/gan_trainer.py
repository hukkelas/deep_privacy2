import atexit
from collections import defaultdict
import logging
import typing
import torch
import time
from dp2.utils import vis_utils
from dp2 import utils
from tops import logger, checkpointer
import tops
from easydict import EasyDict


def accumulate_gradients(params, fp16_ddp_accumulate):
    if len(params) == 0:
        return
    params = [param for param in params if param.grad is not None]
    flat = torch.cat([param.grad.flatten() for param in params])
    orig_dtype = flat.dtype
    if tops.world_size() > 1:
        if fp16_ddp_accumulate:
            flat = flat.half() / tops.world_size()
        else:
            flat /= tops.world_size()
        torch.distributed.all_reduce(flat)
        flat = flat.to(orig_dtype)
    grads = flat.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.shape)


def accumulate_buffers(module: torch.nn.Module):
    buffers = [buf for buf in module.buffers()]
    if len(buffers) == 0:
        return
    flat = torch.cat([buf.flatten() for buf in buffers])
    if tops.world_size() > 1:
        torch.distributed.all_reduce(flat)
        flat /= tops.world_size()
    bufs = flat.split([buf.numel() for buf in buffers])
    for old, new in zip(buffers, bufs):
        old.copy_(new.reshape(old.shape), non_blocking=True)


def check_ddp_consistency(module):
    if tops.world_size() == 1:
        return
    assert isinstance(module, torch.nn.Module)
    assert isinstance(module, torch.nn.Module)
    params_buffs = list(module.named_parameters()) + list(module.named_buffers())
    for name, tensor in params_buffs:
        fullname = type(module).__name__ + '.' + name
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname


class AverageMeter():
    def __init__(self) -> None:
        self.to_log = dict()
        self.n = defaultdict(int)
        pass

    @torch.no_grad()
    def update(self, values: dict):
        for key, value in values.items():
            self.n[key] += 1
            if key in self.to_log:
                self.to_log[key] += value.mean().detach()
            else:
                self.to_log[key] = value.mean().detach()

    def get_average(self):
        return {key: value / self.n[key] for key, value in self.to_log.items()}


class GANTrainer:

    def __init__(
            self,
            G: torch.nn.Module,
            D: torch.nn.Module,
            G_EMA: torch.nn.Module,
            D_optim: torch.optim.Optimizer,
            G_optim: torch.optim.Optimizer,
            dl_train: typing.Iterator,
            dl_val: typing.Iterable,
            scaler_D: torch.cuda.amp.GradScaler,
            scaler_G: torch.cuda.amp.GradScaler,
            ims_per_log: int,
            max_images_to_train: int,
            loss_handler,
            ims_per_val: int,
            evaluate_fn,
            batch_size: int,
            broadcast_buffers: bool,
            fp16_ddp_accumulate: bool,
            save_state: bool,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.G = G
        self.D = D
        self.G_EMA = G_EMA
        self.D_optim = D_optim
        self.G_optim = G_optim
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.scaler_D = scaler_D
        self.scaler_G = scaler_G
        self.loss_handler = loss_handler
        self.max_images_to_train = max_images_to_train
        self.images_per_val = ims_per_val
        self.images_per_log = ims_per_log
        self.evaluate_fn = evaluate_fn
        self.batch_size = batch_size
        self.broadcast_buffers = broadcast_buffers
        self.fp16_ddp_accumulate = fp16_ddp_accumulate

        self.train_state = EasyDict(
            next_log_step=0,
            next_val_step=ims_per_val,
            total_time=0
        )

        checkpointer.register_models(dict(
            generator=G, discriminator=D, EMA_generator=G_EMA,
            D_optimizer=D_optim,
            G_optimizer=G_optim,
            train_state=self.train_state,
            scaler_D=self.scaler_D,
            scaler_G=self.scaler_G
        ))
        if checkpointer.has_checkpoint():
            checkpointer.load_registered_models()
            logger.log(f"Resuming training from: global step: {logger.global_step()}")
        else:
            logger.add_dict({
                "stats/discriminator_parameters": tops.num_parameters(self.D),
                "stats/generator_parameters": tops.num_parameters(self.G),
            }, commit=False)
        if save_state:
            # If the job is unexpectedly killed, there could be a mismatch between previously saved checkpoint and the current checkpoint.
            atexit.register(checkpointer.save_registered_models)

        self._ims_per_log = ims_per_log

        self.to_log = AverageMeter()
        self.trainable_params_D = [param for param in self.D.parameters() if param.requires_grad]
        self.trainable_params_G = [param for param in self.G.parameters() if param.requires_grad]
        logger.add_dict({
            "stats/discriminator_trainable_parameters": sum(p.numel() for p in self.trainable_params_D),
            "stats/generator_trainable_parameters": sum(p.numel() for p in self.trainable_params_G),
        }, commit=False, level=logging.INFO)
        check_ddp_consistency(self.D)
        check_ddp_consistency(self.G)
        check_ddp_consistency(self.G_EMA.generator)

    def train_loop(self):
        self.log_time()
        while logger.global_step() <= self.max_images_to_train:
            batch = next(self.dl_train)
            self.G_EMA.update_beta()
            self.to_log.update(self.step_D(batch))
            self.to_log.update(self.step_G(batch))
            self.G_EMA.update(self.G)

            if logger.global_step() >= self.train_state.next_log_step:
                to_log = {f"loss/{key}": item.item() for key, item in self.to_log.get_average().items()}
                to_log.update({"amp/grad_scale_G": self.scaler_G.get_scale()})
                to_log.update({"amp/grad_scale_D": self.scaler_D.get_scale()})
                self.to_log = AverageMeter()
                logger.add_dict(to_log, commit=True)
                self.train_state.next_log_step += self.images_per_log
            if self.scaler_D.get_scale() < 1e-8 or self.scaler_G.get_scale() < 1e-8:
                print("Stopping training as gradient scale < 1e-8")
                logger.log("Stopping training as gradient scale < 1e-8")
                break

            if logger.global_step() >= self.train_state.next_val_step:
                self.evaluate()
                self.log_time()
                self.save_images()
                self.train_state.next_val_step += self.images_per_val
            logger.step(self.batch_size*tops.world_size())
        logger.log(f"Reached end of training at step {logger.global_step()}.")
        checkpointer.save_registered_models()

    def estimate_ims_per_hour(self):
        batch = next(self.dl_train)
        n_ims = int(100e3)
        n_steps = int(n_ims / (self.batch_size * tops.world_size()))
        n_ims = n_steps * self.batch_size * tops.world_size()
        for i in range(10):  # Warmup
            self.G_EMA.update_beta()
            self.step_D(batch)
            self.step_G(batch)
            self.G_EMA.update(self.G)
        start_time = time.time()
        for i in utils.tqdm_(list(range(n_steps))):
            self.G_EMA.update_beta()
            self.step_D(batch)
            self.step_G(batch)
            self.G_EMA.update(self.G)
        total_time = time.time() - start_time
        ims_per_sec = n_ims / total_time
        ims_per_hour = ims_per_sec * 60*60
        ims_per_day = ims_per_hour * 24
        logger.log(f"Images per hour: {ims_per_hour/1e6:.3f}M")
        logger.log(f"Images per day: {ims_per_day/1e6:.3f}M")
        import math
        ims_per_4_day = int(math.ceil(ims_per_day / tops.world_size() * 4))
        logger.log(f"Images per 4 days: {ims_per_4_day}")
        logger.add_dict({
            "stats/ims_per_day": ims_per_day,
            "stats/ims_per_4_day": ims_per_4_day
        })

    def log_time(self):
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
            self.last_time_step = logger.global_step()
            return
        n_images = logger.global_step() - self.last_time_step
        if n_images == 0:
            return
        n_secs = time.time() - self.start_time
        n_ims_per_sec = n_images / n_secs
        training_time_hours = n_secs / 60 / 60
        self.train_state.total_time += training_time_hours
        remaining_images = self.max_images_to_train - logger.global_step()
        remaining_time = remaining_images / n_ims_per_sec / 60 / 60
        logger.add_dict({
            "stats/n_ims_per_sec": n_ims_per_sec,
            "stats/total_traing_time_hours": self.train_state.total_time,
            "stats/remaining_time_hours": remaining_time
        })
        self.last_time_step = logger.global_step()
        self.start_time = time.time()

    def save_images(self):
        dl_val = iter(self.dl_val)
        batch = next(dl_val)
        # TRUNCATED visualization
        ims_to_log = 8
        self.G_EMA.eval()
        z = self.G.get_z(batch["img"])
        fakes_truncated = self.G_EMA.sample(**batch, truncation_value=0)["img"]
        fakes_truncated = utils.denormalize_img(fakes_truncated).mul(255).byte()[:ims_to_log].cpu()
        if "__key__" in batch:
            batch.pop("__key__")
        real = vis_utils.visualize_batch(**tops.to_cpu(batch))[:ims_to_log]
        to_vis = torch.cat((real, fakes_truncated))
        logger.add_images("images/truncated", to_vis, nrow=2)

        # Diverse images
        ims_diverse = 3
        batch = next(dl_val)
        to_vis = []

        for i in range(ims_diverse):
            z = self.G.get_z(batch["img"])[:1].repeat(batch["img"].shape[0], 1)
            fakes = utils.denormalize_img(self.G_EMA(**batch, z=z)["img"]).mul(255).byte()[:ims_to_log].cpu()
            to_vis.append(fakes)
        if "__key__" in batch:
            batch.pop("__key__")
        reals = vis_utils.visualize_batch(**tops.to_cpu(batch))[:ims_to_log]
        to_vis.insert(0, reals)
        to_vis = torch.cat(to_vis)
        logger.add_images("images/diverse", to_vis, nrow=ims_diverse+1)

        self.G_EMA.train()
        pass

    def evaluate(self):
        logger.log("Stating evaluation.")
        self.G_EMA.eval()
        try:
            checkpointer.save_registered_models(max_keep=3)
        except Exception:
            logger.log("Could not save checkpoint.")
        if self.broadcast_buffers:
            check_ddp_consistency(self.G)
            check_ddp_consistency(self.D)
        metrics = self.evaluate_fn(generator=self.G_EMA, dataloader=self.dl_val)
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        logger.add_dict(metrics, level=logger.logger.INFO)

    def step_D(self, batch):
        utils.set_requires_grad(self.trainable_params_D, True)
        utils.set_requires_grad(self.trainable_params_G, False)
        tops.zero_grad(self.D)
        loss, to_log = self.loss_handler.D_loss(batch, grad_scaler=self.scaler_D)
        with torch.autograd.profiler.record_function("D_step"):
            self.scaler_D.scale(loss).backward()
            accumulate_gradients(self.trainable_params_D, fp16_ddp_accumulate=self.fp16_ddp_accumulate)
            if self.broadcast_buffers:
                accumulate_buffers(self.D)
                accumulate_buffers(self.G)
            # Step will not unscale if unscale is called previously.
            self.scaler_D.step(self.D_optim)
            self.scaler_D.update()
        utils.set_requires_grad(self.trainable_params_D, False)
        utils.set_requires_grad(self.trainable_params_G, False)
        return to_log

    def step_G(self, batch):
        utils.set_requires_grad(self.trainable_params_D, False)
        utils.set_requires_grad(self.trainable_params_G, True)
        tops.zero_grad(self.G)
        loss, to_log = self.loss_handler.G_loss(batch, grad_scaler=self.scaler_G)
        with torch.autograd.profiler.record_function("G_step"):
            self.scaler_G.scale(loss).backward()
            accumulate_gradients(self.trainable_params_G, fp16_ddp_accumulate=self.fp16_ddp_accumulate)
            if self.broadcast_buffers:
                accumulate_buffers(self.G)
                accumulate_buffers(self.D)
            self.scaler_G.step(self.G_optim)
            self.scaler_G.update()
        utils.set_requires_grad(self.trainable_params_D, False)
        utils.set_requires_grad(self.trainable_params_G, False)
        return to_log
