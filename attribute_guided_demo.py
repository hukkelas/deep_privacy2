from collections import defaultdict
import gradio
import click
import numpy as np
import torch
import cv2
from PIL import Image
from dp2 import utils
from tops.config import instantiate
import tops
import gradio.inputs
from stylemc import get_and_cache_direction, get_stylesW, init_affine_modules


class GuidedDemo:
    def __init__(self, anonymizer, cfg_face) -> None:
        self.anonymizer = anonymizer
        assert sum([x is not None for x in list(anonymizer.generators.values())]) == 1
        self.generator = [x for x in list(anonymizer.generators.values()) if x is not None][0]
        cfg = list(anonymizer.generator_cfgs.values())[0]
        cfg.train.batch_size = 1
        self.dl = instantiate(cfg.data.val.loader)
        self.cache_dir = cfg.output_dir
        self.precompute_edits()
        self.is_first = True
        
    def precompute_edits(self):
        self.precomputed_edits = set()
        for edit in self.precomputed_edits:
            get_and_cache_direction(self.cache_dir, self.dl, self.generator, edit)
        if self.cache_dir.joinpath("stylemc_cache").is_dir():
            for path in self.cache_dir.joinpath("stylemc_cache").iterdir():
                text_prompt = path.stem.replace("_", " ")
                self.precomputed_edits.add(text_prompt)
                print(text_prompt)
        self.edits = defaultdict(defaultdict)

    def anonymize(self, img, show_boxes: bool, current_box_idx: int, current_styles, current_boxes, update_identity, edits, cache_id=None):
        if not isinstance(img, torch.Tensor):
            img, cache_id = pil2torch(img)
            img = tops.to_cuda(img)

        current_box_idx = current_box_idx % len(current_boxes)
        edited_styles = [s.clone() for s in current_styles]
        for face_idx, face_edits in edits.items():
            for prompt, strength in face_edits.items():
                direction = get_and_cache_direction(self.cache_dir, self.dl, self.generator, prompt)
                edited_styles[int(face_idx)] += direction * strength
            update_identity[int(face_idx)] = True
        assert img.dtype == torch.uint8
        img = self.anonymizer(
            img, truncation_value=0,
            multi_modal_truncation=True, amp=True,
            cache_id=cache_id,
            all_styles=iter(edited_styles),
            update_identity=update_identity)
        update_identity = [True for i in range(len(update_identity))]
        img = utils.im2numpy(img)
        if show_boxes:
            x0, y0, x1, y1 = [int(_) for _ in current_boxes[int(current_box_idx)]]
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
        return img, update_identity

    def update_image(self, img, show_boxes):
        img, cache_id = pil2torch(img)
        img = tops.to_cuda(img)
        det = self.anonymizer.detector.forward_and_cache(img, cache_id, load_cache=True)[0]
        current_styles = []
        for i in range(len(det)):
            # Need to do forward pass to register all affine modules.
            batch = det.get_crop(i, img)
            batch["condition"] = batch["img"].float()
            if self.multi_modal_truncation:
                w = self.generator.style_net.multi_modal_truncate(0, n=1)
            else:
                w = self.generator.style_net.get_truncated(1, n=1)
            if self.is_first:
                init_affine_modules(self.generator, batch)
            self.is_first = False
            s = get_stylesW(w) 
            current_styles.append(s)
        update_identity = [True for i in range(len(det))]
        current_boxes = np.array(det.boxes)
        edits = defaultdict(defaultdict)
        cur_face_idx = -1 % len(current_boxes)
        img, update_identity = self.anonymize(img, show_boxes, cur_face_idx, current_styles, current_boxes, update_identity, edits, cache_id=cache_id)
        return img, current_styles, current_boxes, update_identity, edits, cur_face_idx

    def change_face(self, change, cur_face_idx, current_boxes, input_image, show_boxes, current_styles, update_identity, edits):
        cur_face_idx = (cur_face_idx+change) % len(current_boxes)
        img, update_identity = self.anonymize(input_image, show_boxes, cur_face_idx, current_styles, current_boxes, update_identity, edits)
        return img, update_identity, cur_face_idx

    def add_style(self, face_idx: int, prompt: str, strength: float, input_image, show_boxes, current_styles, current_boxes, update_identity, edits):
        face_idx = face_idx % len(current_boxes)
        edits[face_idx][prompt] = strength
        img, update_identity = self.anonymize(input_image, show_boxes, face_idx, current_styles, current_boxes, update_identity, edits)
        return img, update_identity, edits

    def switch_mmt(self, value):
        self.multi_modal_truncation = value
    
    def setup_interface(self):
        current_styles = gradio.State()
        current_boxes = gradio.State(None)
        update_identity = gradio.State([])
        edits = gradio.State([])
        with gradio.Row():
            input_image = gradio.Image(
                type="pil", label="Upload your image or try the example below!", source="webcam")
            output_image = gradio.Image(type="numpy", label="Output")
        with gradio.Row():
            update_btn = gradio.Button("Update Anonymization").style(full_width=True)
            mmt_btn = gradio.Checkbox(label="Multi Modal anonymization", value=True)
        
        with gradio.Row():
            show_boxes = gradio.Checkbox(value=True, label="Show Selected")
            cur_face_idx = gradio.Number(value=-1,label="Current", interactive=False)
            previous = gradio.Button("Previous Person")
            next_ = gradio.Button("Next Person")
        with gradio.Row():
            text_prompt = gradio.Textbox(
                placeholder=" | ".join(list(self.precomputed_edits)),
                label="Text Prompt for Edit")
            edit_strength = gradio.Slider(0, 5, step=.01)
            add_btn = gradio.Button("Add Edit")
            add_btn.click(self.add_style, inputs=[cur_face_idx, text_prompt, edit_strength, input_image, show_boxes, current_styles, current_boxes, update_identity, edits], outputs=[output_image, update_identity, edits])
        mmt_btn.change(self.switch_mmt, inputs=[mmt_btn], outputs=[])
        self.multi_modal_truncation = mmt_btn.value
        update_btn.click(self.update_image, inputs=[input_image, show_boxes], outputs=[output_image, current_styles, current_boxes, update_identity, edits, cur_face_idx])
        input_image.change(self.update_image, inputs=[input_image, show_boxes], outputs=[output_image, current_styles, current_boxes, update_identity, edits, cur_face_idx])
        previous.click(self.change_face, inputs=[gradio.State(-1), cur_face_idx, current_boxes, input_image, show_boxes, current_styles, update_identity, edits], outputs=[output_image, update_identity, cur_face_idx])
        next_.click(self.change_face, inputs=[gradio.State(1), cur_face_idx, current_boxes, input_image, show_boxes, current_styles, update_identity, edits], outputs=[output_image, update_identity, cur_face_idx])

        show_boxes.change(self.anonymize, inputs=[input_image, show_boxes, cur_face_idx, current_styles, current_boxes, update_identity, edits], outputs=[output_image, update_identity])


def pil2torch(img: Image.Image):
    img = img.convert("RGB")
    img = np.array(img)
    img = np.rollaxis(img, 2)
    return torch.from_numpy(img), None

@click.command()
@click.argument("config_path")
def main(config_path):
    cfg_face = utils.load_config(config_path)
    anonymizer_face = instantiate(cfg_face.anonymizer, load_cache=False)
    anonymizer_face.initialize_tracker(fps=1)


    with gradio.Blocks() as demo:
        gradio.Markdown("# <center> DeepPrivacy2 - Realistic Image Anonymization </center>")
        gradio.Markdown("### <center> Håkon Hukkelås, Rudolf Mester, Frank Lindseth </center>")
        with gradio.Tab("Text-Guided Anonymization"):
            GuidedDemo(anonymizer_face, cfg_face).setup_interface()


    demo.launch()
main()