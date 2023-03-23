from collections import defaultdict
import gradio
import numpy as np
import torch
import cv2
from PIL import Image
from dp2 import utils
from tops.config import instantiate
import tops
import gradio.inputs
from stylemc import get_and_cache_direction, get_styles
from sg3_torch_utils.ops import grid_sample_gradfix, bias_act, upfirdn2d

grid_sample_gradfix.enabled = False
bias_act.enabled = False
upfirdn2d.enabled = False


class GuidedDemo:
    def __init__(self, face_anonymizer, cfg_face, multi_modal_truncation, truncation_value) -> None:
        self.anonymizer = face_anonymizer
        self.multi_modal_truncation = multi_modal_truncation
        self.truncation_value = truncation_value
        assert sum([x is not None for x in list(face_anonymizer.generators.values())]) == 1
        self.generator = [x for x in list(face_anonymizer.generators.values()) if x is not None][0]
        face_G_cfg = utils.load_config(cfg_face.anonymizer.face_G_cfg)
        face_G_cfg.train.batch_size = 1
        self.dl = instantiate(face_G_cfg.data.val.loader)
        self.cache_dir = face_G_cfg.output_dir
        self.precompute_edits()

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
            img, truncation_value=self.truncation_value,
            multi_modal_truncation=self.multi_modal_truncation, amp=True,
            cache_id=cache_id,
            all_styles=edited_styles,
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
            s = get_styles(
                np.random.randint(0, 999999), self.generator,
                None, truncation_value=self.truncation_value)
            current_styles.append(s)
        update_identity = [True for i in range(len(det))]
        current_boxes = np.array(det.boxes)
        edits = defaultdict(defaultdict)
        cur_face_idx = -1 % len(current_boxes)
        img, update_identity = self.anonymize(
            img, show_boxes, cur_face_idx,
            current_styles, current_boxes, update_identity, edits, cache_id=cache_id)
        return img, current_styles, current_boxes, update_identity, edits, cur_face_idx

    def change_face(self, change, cur_face_idx, current_boxes, input_image, show_boxes, current_styles, update_identity, edits):
        cur_face_idx = (cur_face_idx + change) % len(current_boxes)
        img, update_identity = self.anonymize(
            input_image, show_boxes, cur_face_idx,
            current_styles, current_boxes, update_identity, edits)
        return img, update_identity, cur_face_idx

    def add_style(self, face_idx: int, prompt: str, strength: float, input_image, show_boxes, current_styles, current_boxes, update_identity, edits):
        face_idx = face_idx % len(current_boxes)
        edits[face_idx][prompt] = strength
        img, update_identity = self.anonymize(
            input_image, show_boxes, face_idx,
            current_styles, current_boxes, update_identity, edits)
        return img, update_identity, edits

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
        with gradio.Row():
            show_boxes = gradio.Checkbox(value=True, label="Show Selected")
            cur_face_idx = gradio.Number(value=-1, label="Current", interactive=False)
            previous = gradio.Button("Previous Person")
            next_ = gradio.Button("Next Person")
        with gradio.Row():
            text_prompt = gradio.Textbox(
                placeholder=" | ".join(list(self.precomputed_edits)),
                label="Text Prompt for Edit")
            edit_strength = gradio.Slider(0, 5, step=.01)
            add_btn = gradio.Button("Add Edit")
            add_btn.click(
                self.add_style,
                inputs=[cur_face_idx, text_prompt, edit_strength, input_image, show_boxes,current_styles, current_boxes, update_identity, edits],
                outputs=[output_image, update_identity, edits])
        update_btn.click(
            self.update_image,
            inputs=[input_image, show_boxes],
            outputs=[output_image, current_styles, current_boxes, update_identity, edits, cur_face_idx])
        input_image.change(
            self.update_image,
            inputs=[input_image, show_boxes],
            outputs=[output_image, current_styles, current_boxes, update_identity, edits, cur_face_idx])
        previous.click(
            self.change_face,
            inputs=[gradio.State(-1), cur_face_idx, current_boxes, input_image, show_boxes, current_styles, update_identity, edits],
            outputs=[output_image, update_identity, cur_face_idx])
        next_.click(
            self.change_face,
            inputs=[gradio.State(1), cur_face_idx, current_boxes, input_image, show_boxes,current_styles, update_identity, edits],
            outputs=[output_image, update_identity, cur_face_idx])
        show_boxes.change(
            self.anonymize,
            inputs=[input_image, show_boxes, cur_face_idx, current_styles, current_boxes, update_identity, edits],
            outputs=[output_image, update_identity])


class WebcamDemo:

    def __init__(self, anonymizer) -> None:
        self.anonymizer = anonymizer
        with gradio.Row():
            input_image = gradio.Image(type="pil", source="webcam", streaming=True)
            output_image = gradio.Image(type="numpy", label="Output")
        with gradio.Row():
            truncation_value = gradio.Slider(0, 1, value=0, step=0.01)
            truncation = gradio.Radio(["Multi-modal truncation", "Unimodal truncation"], value="Unimodal truncation")
        with gradio.Row():
            visualize_det = gradio.Checkbox(value=False, label="Show Detections")
            track = gradio.Checkbox(value=False, label="Track detections (samples same latent variable per track)")
        input_image.stream(
            self.anonymize,
            inputs=[input_image, visualize_det, truncation_value,truncation, track, gradio.Variable(False)],
            outputs=[output_image])
        self.track = True

    def anonymize(self, img: Image, visualize_detection: bool, truncation_value, truncation_type, track, reset_track):
        if reset_track:
            self.anonymizer.reset_tracker()
        mmt = truncation_type == "Multi-modal truncation"
        img, cache_id = pil2torch(img)
        img = tops.to_cuda(img)
        self.anonymizer
        if visualize_detection:
            img = self.anonymizer.visualize_detection(img, cache_id=cache_id)
        else:
            img = self.anonymizer(
                img,
                truncation_value=truncation_value,
                multi_modal_truncation=mmt,
                amp=True,
                cache_id=cache_id,
                track=track)
        img = utils.im2numpy(img)
        return img


class ExampleDemo(WebcamDemo):

    def __init__(self, anonymizer) -> None:
        self.anonymizer = anonymizer
        with gradio.Row():
            input_image = gradio.Image(type="pil", source="webcam")
            output_image = gradio.Image(type="numpy", label="Output")
        with gradio.Row():
            update_btn = gradio.Button("Update Anonymization").style(full_width=True)
            resample = gradio.Button("Resample Latent Variables").style(full_width=True)
        with gradio.Row():
            truncation_value = gradio.Slider(0, 1, value=0, step=0.01)
            truncation = gradio.Radio(["Multi-modal truncation", "Unimodal truncation"], value="Unimodal truncation")
        visualize_det = gradio.Checkbox(value=False, label="Show Detections")
        visualize_det.change(
            self.anonymize,
            inputs=[input_image, visualize_det, truncation_value, truncation, gradio.Variable(True), gradio.Variable(False)],
            outputs=[output_image])
        gradio.Examples(
            ["media/erling.jpg", "media/regjeringen.jpg"], inputs=[input_image]
        )

        update_btn.click(
            self.anonymize,
            inputs=[input_image, visualize_det, truncation_value, truncation, gradio.Variable(True), gradio.Variable(False)],
            outputs=[output_image])
        resample.click(
            self.anonymize,
            inputs=[input_image, visualize_det, truncation_value, truncation, gradio.Variable(True), gradio.Variable(True)],
            outputs=[output_image])
        input_image.change(
            self.anonymize,
            inputs=[input_image, visualize_det, truncation_value, truncation, gradio.Variable(False), gradio.Variable(True)],
            outputs=[output_image])
        self.track = False
        self.truncation_value = truncation_value


class Information:

    def __init__(self) -> None:
        gradio.Markdown("## <center> Face Anonymization Architecture </center>")
        gradio.Markdown("---")
        gradio.Image(value="media/overall_architecture.png")
        gradio.Markdown("## <center> Full-Body Anonymization Architecture </center>")
        gradio.Markdown("---")
        gradio.Image(value="media/full_body.png")
        gradio.Markdown("### <center> Generative Adversarial Networks </center>")
        gradio.Markdown("---")
        gradio.Image(value="media/gan_architecture.png")


def pil2torch(img: Image.Image):
    img = img.convert("RGB")
    img = np.array(img)
    img = np.rollaxis(img, 2)
    return torch.from_numpy(img), None
