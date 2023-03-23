import gradio
from dp2 import utils
from tops.config import instantiate
import gradio.inputs
from gradio_demos.modules import ExampleDemo, WebcamDemo

cfg_face = utils.load_config("configs/anonymizers/face.py")
anonymizer_face = instantiate(cfg_face.anonymizer, load_cache=False)
print(anonymizer_face.detector)
anonymizer_face.initialize_tracker(fps=1)


with gradio.Blocks() as demo:
    gradio.Markdown("# <center> DeepPrivacy2 - Realistic Image Anonymization </center>")
    gradio.Markdown("### <center> Håkon Hukkelås, Rudolf Mester, Frank Lindseth </center>")
    with gradio.Tab("Face Anonymization"):
        ExampleDemo(anonymizer_face)
    with gradio.Tab("Live Webcam"):
        WebcamDemo(anonymizer_face)

demo.launch()

