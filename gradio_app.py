import random
import os
from argparse import ArgumentParser

import gradio as gr
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from dotenv import load_dotenv
from PIL import Image

from gqvr.model.generator import SD2Enhancer
# from gqvr.utils.captioner import GPTCaptioner


load_dotenv()
error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--local", action="store_true")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--internvl_caption", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

max_size = (512, 512) 

if args.internvl_caption:
    captioner = None
to_tensor = transforms.ToTensor()

config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        device=args.device,
    )
    model.init_models()
else:
    raise ValueError(config.base_model_type)


def process(
    image,
    image_lq,
    prompt,
    onlyVAE_output=False,
    upscale=1,
    seed=-1, 
    progress=gr.Progress(track_tqdm=True),):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    assert image or image_lq, "Please provide at least one of the two inputs: GT or SPAD image."
    if image:
        image = image.convert("RGB")
    if image_lq:
        image_lq = image_lq.convert("RGB")
    
    # Check image size
    out_w, out_h = tuple(int(x * upscale) for x in image.size)
    if out_w * out_h > max_size[0] * max_size[1]:
        return error_image, (
            "Failed: The requested resolution exceeds the maximum pixel limit. "
            f"Your requested resolution is ({out_h}, {out_w}). "
            f"The maximum allowed pixel count is {max_size[0]} x {max_size[1]} "
            f"= {max_size[0] * max_size[1]} :("
        )
        # TODO: Later crop down (while preserving aspect ratio) instead of just erroring out.
        
    image_tensor = to_tensor(image).unsqueeze(0)
    try:
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            return_type="pil",
            only_vae_output=onlyVAE_output,
            save_Gprocessed_latents=save_Gprocessed_latents,
            fname=fname 
        )[0] 
    except Exception as e:
        return error_image, f"Failed: {e} :("

    return pil_image, f"Success! :)"


def process_burst(photon_cube, prompt, upscale=1, progress=gr.Progress(track_tqdm=True), seed=42):
    raise NotImplementedError


MARKDOWN = """
## gQIR: Generative Quanta Image Restoration

[Project Page](https://aryan-garg.github.io/gqir) | [Paper](https://arxiv.org/abs/2602.20417) | [GitHub](https://github.com/Aryan-Garg/gQIR)

If gQIR is helpful for you, please consider adding a star to our [GitHub Repo](https://github.com/Aryan-Garg/gQIR). Thank you!

```bibtex
@InProceedings{garg_2026_gqir,
    author    = {Garg, Aryan and Ma, Sizhuo and  Gupta, Mohit},
    title     = {gQIR: Generative Quanta Image Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
}
```
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="GT input", type="pil")
            image_lq = gq.Image(label="SPAD input", type="pil")
            prompt = gr.Textbox(label=(
                "Prompt (optional)"
                if args.gpt_caption else "Prompt"
            ))
            onlyVAE_output = gr.Button("Stage 1 (qVAE) output only", value="Only qVAE output")
            upscale = 1
            seed = gr.Number(label="Seed", value=-1)
            run = gr.Button(value="Run")
        with gr.Column():
            result = gr.Image(type="pil", format="png")
            status = gr.Textbox(label="status", interactive=False)
        run.click(
            fn=process,
            inputs=[image, image_lq, prompt, onlyVAE_output, upscale, seed],
            outputs=[result, status],
        )
block.launch(server_name="0.0.0.0" if not args.local else "127.0.0.1", server_port=args.port)