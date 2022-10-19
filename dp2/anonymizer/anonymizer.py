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


def resize_batch(img, mask, maskrcnn_mask, condition, imsize, **kwargs):
    img = F.resize(img, imsize, antialias=True)
    mask = (F.resize(mask, imsize, antialias=True) > 0.99).float()
    maskrcnn_mask = (F.resize(maskrcnn_mask, imsize, antialias=True) > 0.5).float()
    
    condition = img * mask
    return dict(img=img, mask=mask, maskrcnn_mask=maskrcnn_mask, condition=condition)
    

class Anonymizer:

    def __init__(
            self,
            detector,
            load_cache: bool,
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
        self.cur_z_idx = 0

    @torch.no_grad()
    def anonymize_detections(self,
            im, detection, truncation_value: float,
            multi_modal_truncation: bool, amp: bool, z_idx,
            all_styles=None,
            update_identity=None,
            ):
        G = self.generators[type(detection)]
        if G is None:
            return im
        C, H, W = im.shape
        orig_im = im.clone()
        if update_identity is None:
            update_identity = [True for i in range(len(detection))]
        for idx in range(len(detection)):
            if not update_identity[idx]:
                continue
            batch = detection.get_crop(idx, im)
            x0, y0, x1, y1 = batch.pop("boxes")[0]
            batch = {k: tops.to_cuda(v) for k, v in batch.items()}
            batch["img"] = F.normalize(batch["img"].float(), [0.5*255, 0.5*255, 0.5*255], [0.5*255, 0.5*255, 0.5*255])
            batch["img"] = batch["img"].float()
            batch["condition"] = batch["mask"] * batch["img"]
            orig_shape = None
            if G.imsize and batch["img"].shape[-1] != G.imsize[-1] and batch["img"].shape[-2] != G.imsize[-2]:
                orig_shape = batch["img"].shape[-2:]
                batch = resize_batch(**batch, imsize=G.imsize)
            with torch.cuda.amp.autocast(amp):
                if all_styles is not None:
                    anonymized_im = G(**batch, s=iter(all_styles[idx]))["img"]
                elif multi_modal_truncation and hasattr(G, "multi_modal_truncate") and hasattr(G.style_net, "w_centers"):
                    w_indices = None
                    if z_idx is not None:
                        w_indices = [z_idx[idx] % len(G.style_net.w_centers)]
                    anonymized_im = G.multi_modal_truncate(
                        **batch, truncation_value=truncation_value,
                        w_indices=w_indices)["img"]
                else:
                    z = None
                    if z_idx is not None:
                        state = np.random.RandomState(seed=z_idx[idx])
                        z = state.normal(size=(1, G.z_channels))
                        z = tops.to_cuda(torch.from_numpy(z))
                    anonymized_im = G.sample(**batch, truncation_value=truncation_value, z=z)["img"]
            if orig_shape is not None:
                anonymized_im = F.resize(anonymized_im, orig_shape, antialias=True)
            anonymized_im = (anonymized_im+1).div(2).clamp(0, 1).mul(255).round().byte()

            # Resize and denormalize image
            gim = F.resize(anonymized_im[0], (y1-y0, x1-x0), antialias=True)
            mask = F.resize(batch["mask"][0], (y1-y0, x1-x0), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            # Remove padding
            pad = [max(-x0,0), max(-y0,0)]
            pad = [*pad, max(x1-W,0), max(y1-H,0)]
            remove_pad = lambda x: x[...,pad[1]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[2]]
            gim = remove_pad(gim)
            mask = remove_pad(mask)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W), min(y1, H)
            mask = mask.logical_not()[None].repeat(3, 1, 1)
            im[:, y0:y1, x0:x1][mask] = gim[mask]

        return im

    def visualize_detection(self, im: torch.Tensor, cache_id: str = None) -> torch.Tensor:
        all_detections = self.detector.forward_and_cache(im, cache_id, load_cache=self.load_cache)
        for det in all_detections:
            im = det.visualize(im)
        return im

    @torch.no_grad()
    def forward(self, im: torch.Tensor, cache_id: str = None, track=True, **synthesis_kwargs) -> torch.Tensor:
        assert im.dtype == torch.uint8
        im = tops.to_cuda(im)
        all_detections = self.detector.forward_and_cache(im, cache_id, load_cache=self.load_cache)
        if hasattr(self, "tracker") and track:
            [_.pre_process() for _ in all_detections]
            import numpy as np 
            boxes = np.concatenate([_.boxes for _ in all_detections])
            boxes = [Detection(box) for box in boxes]
            self.tracker.step(boxes)
            track_ids = self.tracker.detections_matched_ids
            z_idx = []
            for track_id in track_ids:
                if track_id not in self.track_to_z_idx:
                    self.track_to_z_idx[track_id] = self.cur_z_idx
                    self.cur_z_idx += 1
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

