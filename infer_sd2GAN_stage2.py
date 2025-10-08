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

def process(image,prompt,upscale, save_Gprocessed_latents: bool = False, fname:str = "", seed=310):
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
            only_vae_output=False,
            save_Gprocessed_latents=save_Gprocessed_latents,
            fname=fname 
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


def eval_udm():
    udm_videos = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009"]
    for udm_video in udm_videos:
        if not os.path.exists(f"/mnt/disks/behemoth/datasets/UDM10_video/{udm_video}"):
            raise FileNotFoundError(f"GT video folder for UDM10 video {udm_video} not found.")
        gt_video_path = f"/mnt/disks/behemoth/datasets/UDM10_video/{udm_video}"
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
            gt_img_pil.save(os.path.join(output_dir, f"{udm_video}_gt_{zero_filled_str}.png"))
            lq_img_pil.save(os.path.join(output_dir, f"{udm_video}_lq_{zero_filled_str}.png"))
            reconstructed_pil.save(os.path.join(output_dir, f"{udm_video}_out_{zero_filled_str}.png"))

            if len(used_prompt) > 50:
                with open(os.path.join(output_dir, f"prompt_{zero_filled_str}.txt"), "w") as f:
                    f.write(used_prompt)


def get_mosaic(img):
    """
        Convert a demosaiced RGB image (HxWx3) into an RGGB Bayer mosaic.
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    bayer = np.zeros_like(img)
    bayer_pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])
    if bayer_pattern_type == "RGGB":
        # Red
        bayer[0::2, 0::2, 0] = R[0::2, 0::2]
        # Green
        bayer[0::2, 1::2, 1] = G[0::2, 1::2]
        bayer[1::2, 0::2, 1] = G[1::2, 0::2]
        # Blue
        bayer[1::2, 1::2, 2] = B[1::2, 1::2]
    elif bayer_pattern_type == "GRBG":
        # Red
        bayer[0::2, 1::2, 0] = R[0::2, 1::2]
        # Green 
        bayer[0::2, 0::2, 1] = G[0::2, 0::2]
        bayer[1::2, 1::2, 1] = G[1::2, 1::2]
        # Blue
        bayer[1::2, 0::2, 2] = B[1::2, 0::2]
        
    elif bayer_pattern_type == "BGGR":
        # Blue
        bayer[0::2, 0::2, 2] = B[0::2, 0::2]
        # Green
        bayer[0::2, 1::2, 1] = G[0::2, 1::2]
        bayer[1::2, 0::2, 1] = G[1::2, 0::2]
        # Red
        bayer[1::2, 1::2, 0] = R[1::2, 1::2]
    
    else: # GBRG
        # Green
        bayer[0::2, 0::2, 1] = G[0::2, 0::2]
        bayer[1::2, 1::2, 1] = G[1::2, 1::2]
        # Blue
        bayer[0::2, 1::2, 2] = B[0::2, 1::2]
        # Red
        bayer[1::2, 0::2, 0] = R[1::2, 0::2]
    return bayer

import piq
def compute_full_reference_metrics(gt_img, out_img):
    # PSNR SSIM LPIPS
    
    # print(gt_img.shape, out_img.shape)
    # print(gt_img.max(), out_img.max(), gt_img.min(), out_img.min())
    # print("Full-reference scores:")
    psnr = piq.psnr(out_img, gt_img, data_range=1., reduction='none')
    # print(f"PSNR: {psnr.item():.2f} dB")

    ssim = piq.ssim(out_img, gt_img, data_range=1.) 
    # print(f"SSIM: {ssim.item():.4f}")

    lpips = piq.LPIPS(reduction='none')(out_img, gt_img)
    # print(f"LPIPS: {lpips.item():.4f}")
    return psnr.item(), ssim.item(), lpips.item()

import pyiqa
# from DeQAScore.src import Scorer
def compute_no_reference_metrics(out_img):
    # center crop to 224x224 for no-reference metrics
    _, _, h, w = out_img.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    out_img = out_img[:, :, top:top+224, left:left+224]

    # ManIQA DeQA MUSIQ ClipIQA
    maniqa = pyiqa.create_metric('maniqa', device=torch.device("cuda:1"))
    clipiqa = pyiqa.create_metric('clipiqa', device=torch.device("cuda:1"))
    musiq = pyiqa.create_metric('musiq', device=torch.device("cuda:1"))
    # deqa = Scorer(model_type='deqa', device=torch.device("cuda:1"))

    maniqa_score = maniqa(out_img).item()
    clipiqa_score = clipiqa(out_img).item()
    musiq_score = musiq(out_img).item()
    # deqa_score = deqa.score([Image.fromarray((out_img.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))])[0]

    # print("No-reference scores:")
    # print(f"ManIQA: {maniqa_score:.4f}")
    # print(f"ClipIQA: {clipiqa_score:.4f}")
    # print(f"MUSIQ: {musiq_score:.4f}")
    # print(f"DeQA: {deqa_score:.4f}")
    return maniqa_score, clipiqa_score, musiq_score #, deqa_score


def eval_single_image(gt_image_path, lq_image_path=None, lq_bits=3, color=False):
    if gt_image_path:
        gt_img = Image.open(gt_image_path)
        gt_img = gt_img.convert("RGB")
        gt_img = center_crop_arr(gt_img, 512)  # Center Crop to 512x512

        if lq_image_path == None: # LQ image simulation
            bits = lq_bits
            N = 2**bits - 1
            img_lq_sum = np.zeros_like(gt_img, dtype=np.float32)
            for i in range(N): # 4-bit (2**4 - 1)
                # Bayer:
                if color:
                    img_lq_sum = img_lq_sum + get_mosaic(generate_spc_from_gt(gt_img))
                # DeMosaiced:
                else:
                    img_lq_sum = img_lq_sum + generate_spc_from_gt(gt_img)
            img_lq = img_lq_sum / (1.0*N)

            # Select channel 0 of img_lq and repeat it 3 times to make it 3-channel --- monochrome results
            # img_lq = np.repeat(img_lq[:, :, 0:1], 3, axis=2)
        else:
            img_lq = Image.open(lq_image_path)
    else:
        assert lq_image_path != None, "[!] Come on! Atleast provide the GT OR the lq path dude!"
        img_lq = Image.open(lq_image_path)


    out = process(img_lq, "", upscale=1.0)
    # print(out)
    result = (gt_img, img_lq, out[-1], out[0]) # (GT image, LQ image, prompt, reconstructed image)

    # Save results
    output_dir = "./evaluation/"
    os.makedirs(output_dir, exist_ok=True)

    gt_img_torch = to_tensor(result[0]).unsqueeze(0)
    out_img_torch = to_tensor(result[-1]).unsqueeze(0)

    psnr, ssim, lpips = compute_full_reference_metrics(gt_img_torch, out_img_torch)
    maniqa, clipiqa, musiq = compute_no_reference_metrics(out_img_torch)

    if color:
        gt_img_pil = Image.fromarray(gt_img.astype(np.uint8))
        lq_img_pil = Image.fromarray((img_lq*255).astype(np.uint8))
        reconstructed_pil = result[-1]
    else:
        gt_img_pil = Image.fromarray(gt_img.astype(np.uint8))#.convert('L')
        lq_img_pil = Image.fromarray((img_lq*255).astype(np.uint8))#.convert('L')
        reconstructed_pil = result[-1]#.convert('L')

    gt_img_pil.save(os.path.join(output_dir, f"gt_3bit_{'color' if color else 'mono'}.png"))
    lq_img_pil.save(os.path.join(output_dir, f"lq_3bit_{'color' if color else 'mono'}.png"))
    reconstructed_pil.save(os.path.join(output_dir, f"out_3bit_{'color' if color else 'mono'}.png"))

    # if len(result[-2]) > 50:
    #     with open(os.path.join(output_dir, f"prompt_{}.txt"), "w") as f:
    #         f.write(result[-2])
    return psnr, ssim, lpips, maniqa, clipiqa, musiq



def save_all_video_Gprocessed_latents_to_disk(ds_txt_file):
    # "/home/argar/apgi/gQVR/dataset_txt_files/video_dataset_txt_files/combined_video_dataset.txt"
    print(f"Processing {ds_txt_file}")
    HARDDISK_DIR = "/mnt/disks/behemoth/datasets/"
    # Get all video paths from txt file
    video_paths = []
    with open(ds_txt_file, "r") as fin:
        for line in fin:
            p = line.strip()
            if p:
                p = p.split(" ")
                video_path = p[0]
                video_paths.append(HARDDISK_DIR + video_path[2:])

    for video_path in tqdm(video_paths[::-1]):
        for img_name in sorted(os.listdir(video_path)):
            if os.path.exists(os.path.join(video_path, f"{img_name[:-4]}.pt")) or img_name.endswith(".pt"):
                continue
            image_path = os.path.join(video_path, img_name)
            # print(f"Loading {image_path}")
            gt_image = Image.open(image_path).convert("RGB")
            gt_image = center_crop_arr(gt_image, 512)
            bits = 3
            N = 2**bits - 1
            img_lq_sum = np.zeros_like(gt_image, dtype=np.float32)
            for i in range(N): # 4-bit (2**4 - 1)
                img_lq_sum = img_lq_sum + generate_spc_from_gt(gt_image)
            img_lq = img_lq_sum / (1.0*N)

            try:
                out = process(img_lq, "", upscale= 1.0, save_Gprocessed_latents=True, fname=f"{image_path[:-4]}.pt")
            except Exception as e:
                print(e)
                print(f"[!] Could not save latents for {image_path}  :(")


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--internvl_caption", action="store_true")
    parser.add_argument("--max_size", type=str, default="512,512", help="Comma-seperated image size")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--ds_txt", type=str)
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

    prefix_path = "/media/agarg54/Extreme SSD/"
    psnr_list = []
    ssim_list = []
    lpips_list = []
    maniqa_list = []
    clipiqa_list = []
    musiq_list = []
    with open("/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            gt_image_path = os.path.join(prefix_path, line.strip()[2:])
            metrics = eval_single_image(gt_image_path = gt_image_path, color=config.color)
            psnr_list.append(metrics[0])
            ssim_list.append(metrics[1])
            lpips_list.append(metrics[2])
            maniqa_list.append(metrics[3])
            clipiqa_list.append(metrics[4])
            musiq_list.append(metrics[5])

    with open("./evaluation/mono_full_test_SD21-3bit-Stage2.txt", "w") as f:
        f.write("Overall scores on full_test_set:\n")
        f.write("---------- FR scores ----------\n")
        f.write(f"Average PSNR: {np.mean(psnr_list):.2f} dB\n")
        f.write(f"Average SSIM: {np.mean(ssim_list):.4f}\n")
        f.write(f"Average LPIPS: {np.mean(lpips_list):.4f}\n")
        f.write("---------- NR scores ----------\n")
        f.write(f"Average ManIQA: {np.mean(maniqa_list):.4f}\n")
        f.write(f"Average ClipIQA: {np.mean(clipiqa_list):.4f}\n")
        f.write(f"Average MUSIQ: {np.mean(musiq_list):.4f}\n")

    # save_all_video_Gprocessed_latents_to_disk(args.ds_txt)
        
    
 