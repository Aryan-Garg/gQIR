import random
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms.functional import to_tensor
from accelerate.utils import set_seed
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from gqvr.utils.common import instantiate_from_config
from gqvr.dataset.utils import srgb_to_linearrgb, emulate_spc, center_crop_arr
from gqvr.model.generator import SD2Enhancer


to_tensor = transforms.ToTensor()

def process(image,prompt,upscale,seed=310):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    # Check image size
    if max_size is not None:
        out_w, out_h = tuple(int(x * upscale) for x in image.shape[:2])
        if out_w * out_h > max_size[0] * max_size[1]:
            return (
                "Failed: The requested resolution exceeds the maximum pixel limit. "
                f"Your requested resolution is ({out_h}, {out_w}). "
                f"The maximum allowed pixel count is {max_size[0]} x {max_size[1]} "
                f"= {max_size[0] * max_size[1]} :("
            )
        
    # TODO: Add InternVL-3 Captions 
    if prompt == "auto":
        prompt = ""

    image_tensor = to_tensor(image).unsqueeze(0)
    # print(f"Image shape: {image_tensor.shape}, dtype: {image_tensor.dtype}, device: {image_tensor.device}")
    try:
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            return_type="pil",
        )[0]
    except Exception as e:
        return f"Failed: {e} :("

    return pil_image, f"Used prompt: {prompt}"


def generate_spc_from_gt(img_gt):
    if img_gt is None:
        return None
    img = srgb_to_linearrgb(img_gt / 255.)
    img = emulate_spc(img, 
                      factor=1.0 # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                    )
    return img


################################################# MAIN FUNCTION #################################################
# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--internvl_caption", action="store_true")
parser.add_argument("--max_size", type=str, default="512,512", help="Comma-seperated image size")
parser.add_argument("--device", type=str, default="cuda:5")
args = parser.parse_args()

# Set device
if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device("cpu")
print(f"\nUsing device: {device}\n")


if args.internvl_caption:
    captioner = None

max_size = args.max_size
if max_size is not None:
    max_size = tuple(int(x) for x in max_size.split(","))
    if len(max_size) != 2:
        raise ValueError(f"Invalid max size: {max_size}")
    print(f"Max size set to {max_size}, max pixels: {max_size[0] * max_size[1]}")

# Load configuration
config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        vae_cfg=config.model.vae_cfg,
        device=args.device,
    )
    model.init_models()
else:
    raise ValueError(config.base_model_type)


# Get GT images. TODO: Set BASE PATH to your GT video folder
gt_video_path = "/mnt/disks/behemoth/datasets/UDM10_video/004"
gt_imgs = []
lq_imgs = []
for img_name in tqdm(sorted(os.listdir(gt_video_path)), 
                     total=len(os.listdir(gt_video_path)), 
                     desc="Loading GT images"):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        gt_img = Image.open(os.path.join(gt_video_path, img_name))
        gt_img = gt_img.convert("RGB")
        gt_img = center_crop_arr(gt_img, 512)  # Center Crop to 512x512
        gt_imgs.append(gt_img) # Tx512x512x3 images [0-255]


# LQ image simulation
bits = 3
N = 2**bits - 1
for img_np in gt_imgs:
    img_lq_sum = np.zeros_like(img_np, dtype=np.float32)
    for i in range(N): # 4-bit (2**4 - 1)
        img_lq_sum = img_lq_sum + generate_spc_from_gt(img_np)
    img_lq = img_lq_sum / (1.0*N)
    lq_imgs.append(img_lq)        

results = []
pbar = tqdm(enumerate(zip(gt_imgs, lq_imgs)) , total=len(gt_imgs), desc="Processing images")
for i, (gt_img, lq_img) in pbar:
    pbar.set_description(f"Processing image {i+1}/{len(gt_imgs)}")
    out = process(lq_img, "", upscale=1.0)
    results.append((gt_img, lq_img, out[-1], out[0]))  # (GT image, LQ image, prompt, reconstructed image)

# Save results
output_dir = "./evaluation/"
os.makedirs(output_dir, exist_ok=True)
for i, (gt_img, lq_img, used_prompt, reconstructed) in enumerate(results):
    gt_img_pil = Image.fromarray(gt_img.astype(np.uint8))
    lq_img_pil = Image.fromarray((lq_img*255).astype(np.uint8))
    reconstructed_pil = reconstructed

    zero_filled_str = str(i+1).zfill(3)
    gt_img_pil.save(os.path.join(output_dir, f"gt_{zero_filled_str}.png"))
    lq_img_pil.save(os.path.join(output_dir, f"lq_{zero_filled_str}.png"))
    reconstructed_pil.save(os.path.join(output_dir, f"reconstructed_{zero_filled_str}.png"))

    if len(used_prompt) > 50:
        with open(os.path.join(output_dir, f"prompt_{zero_filled_str}.txt"), "w") as f:
            f.write(used_prompt)