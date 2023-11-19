# Code adapted from: https://github.com/gpastal24/ViTPose-Pytorch
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
import torch
from .backbone import ViT
import torchvision.transforms.functional as F
import torch.nn as nn
import tops

model_large = dict(
    type="TopDown",
    pretrained=None,
    backbone=dict(
        type="ViT",
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.5,
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1024,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=17,
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process="default",
        shift_heatmap=False,
        target_type="GaussianHeatmap",
        modulate_kernel=11,
        use_udp=True,
    ),
)


model_base = dict(
    type="TopDown",
    pretrained=None,
    backbone=dict(
        type="ViT",
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=17,
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)
model_huge = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type="GaussianHeatmap",
        modulate_kernel=11,
        use_udp=True))


class VitPoseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        assert model_name  in ["vit_base", "vit_large", "vit_huge"]
        model = {
            "vit_base": model_base,
            "vit_large": model_large,
            "vit_huge": model_huge
        }[model_name]
        weight_url = {
            "vit_base": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/90235a26-3b8c-427d-a264-c68155abecdcfcfcd8a9-0388-4575-b85b-607d3c0a9b149bef8f0f-a0f9-4662-a561-1b47ba5f1636",
            "vit_large": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/a580a44c-0afd-43ac-a2cb-9956c32b1d1a78c51ecb-81bb-4345-8710-13904cb9dbbe0703db2d-8534-42e0-ac4d-518ab51fe7db",
            "vit_huge": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/a33b6ada-4d2f-4ef7-8f83-b33f58b69f5b2a62e181-2131-467d-a900-027157a08571d761fad4-785b-4b84-8596-8932c7857e44"
        }[model_name]
        file_name = {
            "vit_base":  "vit-b-multi-coco-595b5e128b.pth",
            "vit_large": "vit-l-multi-coco-9475d27cec.pth",
            "vit_huge":  "vit-h-multi-coco-dbc06d4337.pth",
        }[model_name]
        # Set check_hash to true if you suspect a download error.
        weight_path = tops.download_file(
            weight_url, file_name=file_name, check_hash=False)

        self.keypoint_head = tops.to_cuda(TopdownHeatmapSimpleHead(
            in_channels=model["keypoint_head"]["in_channels"],
            out_channels=model["keypoint_head"]["out_channels"],
            num_deconv_filters=model["keypoint_head"]["num_deconv_filters"],
            num_deconv_kernels=model["keypoint_head"]["num_deconv_kernels"],
            num_deconv_layers=model["keypoint_head"]["num_deconv_layers"],
            extra=model["keypoint_head"]["extra"],
        ))
        # print(head)
        self.backbone = tops.to_cuda(ViT(
            img_size=model["backbone"]["img_size"],
            patch_size=model["backbone"]["patch_size"],
            embed_dim=model["backbone"]["embed_dim"],
            depth=model["backbone"]["depth"],
            num_heads=model["backbone"]["num_heads"],
            ratio=model["backbone"]["ratio"],
            mlp_ratio=model["backbone"]["mlp_ratio"],
            qkv_bias=model["backbone"]["qkv_bias"],
            drop_path_rate=model["backbone"]["drop_path_rate"],
        ))
        ckpt = torch.load(weight_path, map_location=tops.get_device())
        self.load_state_dict(ckpt["state_dict"])
        self.backbone.eval()
        self.keypoint_head.eval()

    def forward(self, img: torch.Tensor, boxes_ltrb: torch.Tensor):
        assert img.ndim == 3
        assert img.dtype == torch.uint8
        assert boxes_ltrb.ndim == 2 and boxes_ltrb.shape[1] == 4
        assert boxes_ltrb.dtype == torch.long
        boxes_ltrb = boxes_ltrb.clamp(0)
        padded_boxes = torch.zeros_like(boxes_ltrb)
        images = torch.zeros((len(boxes_ltrb), 3, 256, 192), device=img.device, dtype=torch.float32)

        for i, (x0, y0, x1, y1) in enumerate(boxes_ltrb):
            x1 = min(img.shape[-1], x1)
            y1 = min(img.shape[-2], y1)
            correction_factor = 256 / 192 * (x1 - x0) / (y1 - y0)
            if correction_factor > 1:
                # increase y side
                center = y0 + (y1 - y0) // 2
                length = (y1-y0).mul(correction_factor).round().long()
                y0_new = center - length.div(2).long()
                y1_new = center + length.div(2).long()
                image_crop = img[:, y0:y1, x0:x1]
                # print(y1,y2,x1,x2)
                pad = ((y0_new-y0).abs(), (y1_new-y1).abs())
#                pad = (int(abs(y0_new-y0))), int(abs(y1_new-y1))
                image_crop = torch.nn.functional.pad(image_crop, [*(0, 0), *pad])
                padded_boxes[i] = torch.tensor([x0, y0_new, x1, y1_new])
            else:
                center = x0 + (x1 - x0) // 2
                length = (x1-x0).div(correction_factor).round().long()
                x0_new = center - length.div(2).long()
                x1_new = center + length.div(2).long()
                image_crop = img[:, y0:y1, x0:x1]
                pad = ((x0_new-x0).abs(), (x1_new-x1).abs())
                image_crop = torch.nn.functional.pad(image_crop, [*pad, ])
                padded_boxes[i] = torch.tensor([x0_new, y0, x1_new, y1])
            image_crop = F.resize(image_crop.float(), (256, 192), antialias=True)
            image_crop = F.normalize(image_crop, mean=[0.485*255, 0.456*255,
                                     0.406*255], std=[0.229*255, 0.224*255, 0.225*255])
            images[i] = image_crop

        x = self.backbone(images)
        out = self.keypoint_head(x)
        pts = torch.empty((out.shape[0], out.shape[1], 3), dtype=torch.float32, device=img.device)
        # For each human, for each joint: y, x, confidence
        b, indices = torch.max(out, dim=2)
        b, indices = torch.max(b, dim=2)

        c, indicesc = torch.max(out, dim=3)
        c, indicesc = torch.max(c, dim=2)
        dim1 = torch.tensor(1./64, device=img.device)
        dim2 = torch.tensor(1./48, device=img.device)
        for i in range(0, out.shape[0]):
            pts[i, :, 0] = indicesc[i, :] * dim1 * (padded_boxes[i][3] - padded_boxes[i][1]) + padded_boxes[i][1]
            pts[i, :, 1] = indices[i, :] * dim2 * (padded_boxes[i][2] - padded_boxes[i][0]) + padded_boxes[i][0]
            pts[i, :, 2] = c[i, :]
        pts = pts[:, :, [1, 0, 2]]
        return pts
