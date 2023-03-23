from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
import tops
import torchvision.transforms.functional as F
from motpy import Detection, MultiObjectTracker
from dp2.utils import load_config
from dp2.infer import build_trained_generator
from dp2.detection.structures import CSEPersonDetection, FaceDetection, PersonDetection, VehicleDetection


def load_generator_from_cfg_path(cfg_path: Union[str, Path]):
    cfg = load_config(cfg_path)
    G = build_trained_generator(cfg)
    tops.logger.log(f"Loaded generator from: {cfg_path}")
    return G


class Anonymizer:

    def __init__(
            self,
            detector,
            load_cache: bool = False,
            person_G_cfg: Optional[Union[str, Path]] = None,
            cse_person_G_cfg: Optional[Union[str, Path]] = None,
            face_G_cfg: Optional[Union[str, Path]] = None,
            car_G_cfg: Optional[Union[str, Path]] = None,
    ) -> None:
        self.detector = detector
        self.generators = {k: None for k in [CSEPersonDetection, PersonDetection, FaceDetection, VehicleDetection]}
        self.load_cache = load_cache
        if cse_person_G_cfg is not None:
            self.generators[CSEPersonDetection] = load_generator_from_cfg_path(cse_person_G_cfg)
        if person_G_cfg is not None:
            self.generators[PersonDetection] = load_generator_from_cfg_path(person_G_cfg)
        if face_G_cfg is not None:
            self.generators[FaceDetection] = load_generator_from_cfg_path(face_G_cfg)
        if car_G_cfg is not None:
            self.generators[VehicleDetection] = load_generator_from_cfg_path(car_G_cfg)

    def initialize_tracker(self, fps: float):
        self.tracker = MultiObjectTracker(dt=1/fps)
        self.track_to_z_idx = dict()

    def reset_tracker(self):
        self.track_to_z_idx = dict()

    def forward_G(self,
                  G,
                  batch,
                  multi_modal_truncation: bool,
                  amp: bool,
                  z_idx: int,
                  truncation_value: float,
                  idx: int,
                  all_styles=None):
        batch["img"] = F.normalize(batch["img"].float(), [0.5*255, 0.5*255, 0.5*255], [0.5*255, 0.5*255, 0.5*255])
        batch["img"] = batch["img"].float()
        batch["condition"] = batch["mask"].float() * batch["img"]

        with torch.cuda.amp.autocast(amp):
            z = None
            if z_idx is not None:
                state = np.random.RandomState(seed=z_idx[idx])
                z = state.normal(size=(1, G.z_channels)).astype(np.float32)
                z = tops.to_cuda(torch.from_numpy(z))

            if all_styles is not None:
                anonymized_im = G(**batch, s=iter(all_styles[idx]))["img"]
            elif multi_modal_truncation:
                w_indices = None
                if z_idx is not None:
                    w_indices = [z_idx[idx] % len(G.style_net.w_centers)]
                anonymized_im = G.multi_modal_truncate(
                    **batch, truncation_value=truncation_value,
                    w_indices=w_indices,
                    z=z
                )["img"]
            else:
                anonymized_im = G.sample(**batch, truncation_value=truncation_value, z=z)["img"]
        anonymized_im = (anonymized_im+1).div(2).clamp(0, 1).mul(255)
        return anonymized_im

    @torch.no_grad()
    def anonymize_detections(self,
                             im, detection,
                             update_identity=None,
                             **synthesis_kwargs
                             ):
        G = self.generators[type(detection)]
        if G is None:
            return im
        C, H, W = im.shape
        if update_identity is None:
            update_identity = [True for i in range(len(detection))]
        for idx in range(len(detection)):
            if not update_identity[idx]:
                continue
            batch = detection.get_crop(idx, im)
            x0, y0, x1, y1 = batch.pop("boxes")[0]
            batch = {k: tops.to_cuda(v) for k, v in batch.items()}
            anonymized_im = self.forward_G(G, batch, **synthesis_kwargs, idx=idx)

            gim = F.resize(anonymized_im[0], (y1-y0, x1-x0), interpolation=F.InterpolationMode.BICUBIC, antialias=True)
            mask = F.resize(batch["mask"][0], (y1-y0, x1-x0), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            # Remove padding
            pad = [max(-x0, 0), max(-y0, 0)]
            pad = [*pad, max(x1-W, 0), max(y1-H, 0)]
            def remove_pad(x): return x[..., pad[1]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[2]]

            gim = remove_pad(gim)
            mask = remove_pad(mask) > 0.5
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W), min(y1, H)
            mask = mask.logical_not()[None].repeat(3, 1, 1)

            im[:, y0:y1, x0:x1][mask] = gim[mask].round().clamp(0, 255).byte()
        return im

    def visualize_detection(self, im: torch.Tensor, cache_id: str = None) -> torch.Tensor:
        all_detections = self.detector.forward_and_cache(im, cache_id, load_cache=self.load_cache)
        im = im.cpu()
        for det in all_detections:
            im = det.visualize(im)
        return im

    @torch.no_grad()
    def forward(self, im: torch.Tensor, cache_id: str = None, track=True, detections=None, **synthesis_kwargs) -> torch.Tensor:
        assert im.dtype == torch.uint8
        im = tops.to_cuda(im)
        all_detections = detections
        if detections is None:
            if self.load_cache:
                all_detections = self.detector.forward_and_cache(im, cache_id)
            else:
                all_detections = self.detector(im)
        if hasattr(self, "tracker") and track:
            [_.pre_process() for _ in all_detections]
            boxes = np.concatenate([_.boxes for _ in all_detections])
            boxes = [Detection(box) for box in boxes]
            self.tracker.step(boxes)
            track_ids = self.tracker.detections_matched_ids
            z_idx = []
            for track_id in track_ids:
                if track_id not in self.track_to_z_idx:
                    self.track_to_z_idx[track_id] = np.random.randint(0, 2**32-1)
                z_idx.append(self.track_to_z_idx[track_id])
            z_idx = np.array(z_idx)
            idx_offset = 0

        for detection in all_detections:
            zs = None
            if hasattr(self, "tracker") and track:
                zs = z_idx[idx_offset:idx_offset+len(detection)]
                idx_offset += len(detection)
            im = self.anonymize_detections(im, detection, z_idx=zs, **synthesis_kwargs)

        return im.cpu()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
