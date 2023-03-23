import gradio
from dp2 import utils
from tops.config import instantiate
import gradio.inputs
from gradio_demos.modules import ExampleDemo, WebcamDemo


cfg_body = utils.load_config("configs/anonymizers/FB_cse.py")
anonymizer_body = instantiate(cfg_body.anonymizer, load_cache=False)
anonymizer_body.initialize_tracker(fps=1)


with gradio.Blocks() as demo:
    gradio.Markdown("# <center> DeepPrivacy2 - Realistic Image Anonymization </center>")
    gradio.Markdown("### <center> Håkon Hukkelås, Rudolf Mester, Frank Lindseth </center>")
    with gradio.Tab("Full-Body CSE Anonymization"):
        ExampleDemo(anonymizer_body)
    with gradio.Tab("Full-body CSE Webcam"):
        WebcamDemo(anonymizer_body)


demo.launch()

