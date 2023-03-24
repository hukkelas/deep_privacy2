# Gradio Demos


## Attribute guided anonymization
DeepPrivacy2 allows for controllable anonymization through text prompts by adapting [StyleMC](https://github.com/catlab-team/stylemc).
StyleMC finds global semantically meaningful directions in the GAN latent space by manipulating images towards a given text prompt with a [CLIP](https://github.com/openai/CLIP)-based loss.
<center>
<img src="https://raw.githubusercontent.com/hukkelas/deep_privacy2/master/media/stylemc_example.jpg" alt= “” width="50%">
</center>

The repository includes a [gradio](https://gradio.app/) demo for interactive text-guided anonymization.
To use the demo, first:

1. Download the FDF256 dataset (see [here](training_and_development.md#dataset-setup)). Only the validation set is required.
2. Run the following:
```
python3 attribute_guided_demo.py
```

The script will spin up a local webserver.
