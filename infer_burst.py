import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from argparse import ArgumentParser
import warnings

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm.auto import tqdm
import lpips

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

# Debugging libs: ###############
# import matplotlib.pyplot as plt
import numpy as np
#################################

from gqvr.utils.common import instantiate_from_config, calculate_psnr_pt, to
from gqvr.model.vae import AutoencoderKL

from gqvr.model.core_raft.raft import RAFT
from gqvr.model.core_raft.utils import flow_viz
from gqvr.model.core_raft.utils.utils import InputPadder
from gqvr.model.fusionViT import LightweightHybrid3DFusion

import cv2
from PIL import Image

################################# FLOW STUFF #############################################

def differentiable_warp(x, flow):
    """
    Warp image or feature x according to flow.
    x: [B, C, H, W]
    flow: [B, 2, H, W] (flow in pixels, with flow[:,0] = dx, flow[:,1] = dy)
    """
    B, C, H, W = x.size()
    # Create mesh grid normalized to [-1,1]
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)  # [H, W, 2]

    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

    # Add flow, normalize grid to [-1,1]
    flow = flow.permute(0, 2, 3, 1)
    new_grid = grid + flow
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0

    warped = F.grid_sample(x, new_grid, align_corners=True, padding_mode="border")
    return warped


def compute_flow_magnitude(flow):
    return torch.norm(flow, dim=1, keepdim=True)  # [B, 1, H, W]


def compute_flow_gradients(flow):
    # flow: [B, 2, H, W]
    fx = flow[:, 0:1, :, :]  # horizontal flow
    fy = flow[:, 1:2, :, :]  # vertical flow

    # finite difference gradients (simple Sobel or central differences)
    fx_du = fx[:, :, :, 2:] - fx[:, :, :, :-2]  # d/dx
    fx_dv = fx[:, :, 2:, :] - fx[:, :, :-2, :]  # d/dy

    fy_du = fy[:, :, :, 2:] - fy[:, :, :, :-2]
    fy_dv = fy[:, :, 2:, :] - fy[:, :, :-2, :]

    # pad to original size (pad 1 pixel on each side)
    fx_du = F.pad(fx_du, (1, 1, 0, 0))
    fx_dv = F.pad(fx_dv, (0, 0, 1, 1))
    fy_du = F.pad(fy_du, (1, 1, 0, 0))
    fy_dv = F.pad(fy_dv, (0, 0, 1, 1))

    return fx_du, fx_dv, fy_du, fy_dv


def detect_occlusion(fw_flow, bw_flow, img2):
    """
    fw_flow: forward flow from img1 to img2, [B, 2, H, W]
    bw_flow: backward flow from img2 to img1, [B, 2, H, W]
    img: image tensor (for warping), [B, C, H, W]

    Returns:
        occlusion mask [B, 1, H, W], float tensor (0 or 1 mask)
        warped_img2: img warped back to img1 space by bw_flow
    """

    # Warp forward flow to img2 frame using backward flow
    fw_flow_warped = differentiable_warp(fw_flow, bw_flow)  # [B, 2, H, W]

    # Warp img2 to img1 space using backward flow
    warp_img = differentiable_warp(img2, bw_flow)

    # Forward-backward flow consistency check
    fb_flow_sum = fw_flow_warped + bw_flow  # should be near zero if consistent

    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)  # [B,1,H,W]
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_warped)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    threshold = 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    mask1 = fb_flow_mag > threshold  # bool mask [B,1,H,W]

    # Compute flow gradients for motion boundary detection
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    # Combine masks
    mask = mask1 | mask2  # logical or

    occlusion = mask.float()  # convert to float mask (0 or 1)

    return occlusion, warp_img

##########################################################################################

def print_vram_state(msg=None, logger=None):
    alloc = torch.cuda.memory_allocated() / 1024**3
    cache = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if logger:
        logger.info(
            f"[GPU memory]: {msg}, allocated = {alloc:.2f} GB, "
            f"cached = {cache:.2f} GB, peak = {peak:.2f} GB"
        )
    return alloc, cache, peak


# This is also E^* metric
def compute_flow_loss(pred_frames, gt_frames, raft_model): 
    B, T, C, H, W = pred_frames.shape
    err = 0.0 
    for i in range(T - 1):
        frame1 = pred_frames[:, i, ...]
        frame2 = gt_frames[:, i+1, ...]
        # Compute forward and backward flow using RAFT 
        _, flow_fw = raft_model(frame1, frame2, iters=20, test_mode=True) 
        _, flow_bw = raft_model(frame2, frame1, iters=20, test_mode=True) 
        # Differentiable occlusion mask   
        occ_mask, warp_img2 = detect_occlusion(flow_fw, flow_bw, frame2)  
        noc_mask = 1 - occ_mask 
        diff = (warp_img2 - frame1) * noc_mask 
        diff_squared = diff ** 2 
        N = torch.sum(noc_mask) 
        N = torch.clamp(N, min=1.0) 
        err += torch.sum(diff_squared) / N  
    warping_error = err / (T - 1) 
    return warping_error 


def compute_burst_loss(gt, xhat_lq, lpips_model, scales, loss_mode="gt_perceptual_lsa", z_gt=None, z_fused=None):
    lsa_loss = 0.
    gt_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"lsa_loss": lsa_loss, "perceptual": perceptual_loss, "l2_loss": gt_loss}

    if "lsa" in loss_mode:
        lsa_loss = scales.lsa * F.mse_loss(z_fused, z_gt, reduction="mean")
        loss_dict["lsa_loss"] = lsa_loss.item()

    if "mse" in loss_mode:
        gt_loss = scales.mse * F.mse_loss(xhat_lq.float(), gt.float(), reduction="mean")
        loss_dict["l2_loss"] = gt_loss.item()

    if "perceptual" in loss_mode:
        with torch.cuda.amp.autocast(False):
            perceptual_loss = scales.perceptual * lpips_model( ((xhat_lq*2.) - 1.).float(),  
                                                          ((gt*2.)-1.).float())
        loss_dict["perceptual"] = perceptual_loss.item()

    total_loss = lsa_loss + gt_loss + perceptual_loss
    return total_loss, loss_dict



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


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    weight_dtype = torch.bfloat16
    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.output_dir
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    vae = AutoencoderKL(cfg.model.vae_cfg.ddconfig, cfg.model.vae_cfg.embed_dim)
    # Load quanta VAE
    daVAE = torch.load(cfg.qvae_path, map_location="cpu")
    init_vae = {}
    vae_used = set()
    scratch_vae = vae.state_dict()
    for key in scratch_vae:
        if key not in daVAE:
            print(f"[!] {key} missing in daVAE")
            continue
        # print(f"Found {key} in daVAE. Loading...")
        init_vae[key] = daVAE[key].clone()
        vae_used.add(key)
    vae.load_state_dict(init_vae, strict=True)
    # vae.to(device=device)
    vae_unused = set(daVAE.keys()) - vae_used

    if len(vae_unused) == 0:
        print(f"[+] Loaded qVAE successfully")
    else:
        print(f"[!] VAE keys NOT used: {vae_unused}")

    vae.requires_grad_(False)
    vae.eval().to(device)

    class RAFT_args:
        mixed_precision = True
        small = False
        alternate_corr = True # Reduces VRAM significantly in forward pass
        dropout = False
    raft_args = RAFT_args()

    raft_model = RAFT(raft_args)
    raft_things_dict = torch.load("./pretrained_ckpts/models/raft-things.pth")
    corrected_state_dict = {}
    for k, v in raft_things_dict.items():
        k2 = ".".join(k.split(".")[1:])
        corrected_state_dict[k2] = v

    raft_model.load_state_dict(corrected_state_dict)
    raft_model.eval().requires_grad_(False).to(device)

    # Replace naive averaging with a learned residual solution
    fusion_vit = LightweightHybrid3DFusion()
    fusion_ckpt = torch.load(cfg.fusion_vit_weight_path, map_location="cpu")
    fusion_vit.load_state_dict(fusion_ckpt)
    fusion_vit.eval().requires_grad_(False).to(device)

    # print("[~] Using burst UNet refinement module")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model_path, subfolder="text_encoder", dtype=weight_dtype).to(device)
    text_encoder.eval().requires_grad_(False)

    def encode_prompt(prompt, bs):
        txt_ids = tokenizer(
            [prompt] * bs,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = text_encoder(txt_ids.to(accelerator.device))[0]
        return {"text_embed": text_embed}
     
    
    scheduler = DDPMScheduler.from_pretrained(cfg.base_model_path, subfolder="scheduler")
    ls_burst_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, 
                                                         subfolder="unet", 
                                                         torch_dtype=torch.bfloat16).to(device)
    # Handle lora configuration
    target_modules = cfg.lora_modules
    # print(f"Add lora parameters to {target_modules}")
    G_lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    ls_burst_unet.add_adapter(G_lora_cfg)

    # print(f"Load UNet model weights from {cfg.unet_weight_path}")
    state_dict = torch.load(cfg.unet_weight_path, map_location="cpu", weights_only=False)
    ls_burst_unet.load_state_dict(state_dict, strict=False)
    input_keys = set(state_dict.keys())
    required_keys = set([k for k in ls_burst_unet.state_dict().keys() if "lora" in k])
    missing = required_keys - input_keys
    unexpected = input_keys - required_keys
    assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"
    ls_burst_unet.eval().requires_grad_(False)

    print(f"[~] All fusionVIT model parameters: {sum(p.numel() for p in fusion_vit.parameters()) / 1e6:.2f}M")
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.val)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.val.batch_size,
        num_workers=cfg.dataset.val.num_workers,
        drop_last=False,
        shuffle=False,
        persistent_workers=False, 
        prefetch_factor=2
    )

    batch_transform = instantiate_from_config(cfg.dataset.batch_transform)
    global_step = 0
    psnr_list = []
    ssim_list = []
    lpips_list = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_reconstructed_frames = 0
    # Training loop:
    for ididid, batch in tqdm(enumerate(loader)):
        with torch.inference_mode():
            to(batch, device)
            batch = batch_transform(batch)

            lqs = batch["lqs"].permute(0, 1, 4, 2, 3).to(device) # B T H W C
            gts = batch["gts"].permute(0, 1, 4, 2, 3) # B T C H W to match decoder output
            bs = lqs.size(0)
            T = lqs.size(1)
            center_gt = gts[:, T // 2, ...] # B 3 H W

            # print("Lq details:")
            # print(lqs.shape, lqs.dtype, lqs[:, T//2, ...].min(), lqs[:, T//2, ...].max())

            latents = []
            Y = []
            with torch.no_grad():
                # z_center = vae.encode(center_gt).mode() # B 4 64 64
                for t in range(T):
                    lq_t = lqs[:, t, ...] # B C H W
                    z = vae.encode(lq_t).mode()
                    latents.append(z)
                    lq_hat = vae.decode(z).float() # B 3 H W
                    Y.append(lq_hat)

            # Merge lqs here using RAFT
            center_t = T // 2 
            Y = torch.stack(Y, dim=1) # B T C H W
            # print("Reconstructed lq details:")
            # print(lq_center.shape, lq_center.dtype, lq_center.min(), lq_center.max())
            # # Save lq_center to disk
            # Image.fromarray((((lqs[:,T//2, ...]+1)/2.) * 255.).cpu().numpy().astype('uint8')[0].transpose(1, 2, 0)).save(os.path.join(exp_dir, f"LQ_center_{global_step:06d}.png"))
            # Image.fromarray((lq_center * 255.).cpu().numpy().astype('uint8')[0].transpose(1, 2, 0)).save(os.path.join(exp_dir, f"Y_center_{global_step:06d}.png"))
            # exit()
            flow_vectors = []
            for t in range(T):
                lq_t = Y[:, t, ...] # B C H W
                ls_in = lq_t.float()
                center_in = Y[:, T//2, ...].float()
                if t < center_t:
                    with torch.inference_mode():
                        _, flow_bw = raft_model(center_in, ls_in, iters=20, test_mode=True) # B 2 64 64
                else:
                    with torch.inference_mode():
                        _, flow_bw = raft_model(ls_in, center_in, iters=20, test_mode=True) # B 2 64 64
                # Downsample flow_bw to match latent space size (1, 4, 64, 64) with the highest precision possible
                flow_bw = F.interpolate(flow_bw, size=(64, 64), mode='bilinear', align_corners=True)
                flow_bw[:, 0] *= (64 / cfg.dataset.val.params.out_size)   # scale dx
                flow_bw[:, 1] *= (64 / cfg.dataset.val.params.out_size)   # scale dy
                flow_vectors.append(flow_bw)

            aligned_latents = []

            for t in range(T):
                latent_t = latents[t] # B 4 64 64
                if t == center_t:
                    aligned_latents.append(latent_t)
                    continue
                flow_bw = flow_vectors[t] # B 2 64 64
                warped_latent = differentiable_warp(latent_t, flow_bw) # B 4 64 64
                aligned_latents.append(warped_latent)

            aligned_latents = torch.stack(aligned_latents, dim=1) # B T C H W
            # Average of aligned lqs ---> Changed to learned solution
            merged_latent = fusion_vit(aligned_latents)
            # Sanity check: Send center latent directly
            # merged_latent = latents[T//2]
            del aligned_latents, flow_vectors, latents
            torch.cuda.empty_cache()
            ls_burst_unet.eval()
            with torch.no_grad():
                # For VAE sanity: z = merged_latent
                z_in = (merged_latent * 0.18215).to(weight_dtype) # vae scaling factor
                timesteps = torch.full((bs,), cfg.model_t, dtype=torch.long, device=device)
                eps = ls_burst_unet(
                    z_in,
                    timesteps,
                    encoder_hidden_states=encode_prompt("", bs=bs)["text_embed"],
                ).sample
                z = scheduler.step(eps, cfg.coeff_t, z_in).pred_original_sample
                decoded_refined = (vae.decode(z.float() / 0.18215)).float() # B 3 H W

            decoded_refined = decoded_refined.clamp(0, 1)

            # Save out & gt on disk
            log_gt, log_pred, log_lq = (center_gt[0]+1.)/2., decoded_refined[0], (lqs[0, T//2, ...] + 1.) / 2.
            if cfg.save_imgs:
                log_gt_png = (log_gt * 255.).cpu().numpy().astype('uint8').transpose(1, 2, 0)
                log_pred_png = (log_pred * 255.).cpu().numpy().astype('uint8').transpose(1, 2, 0)
                log_lq_png = (log_lq * 255.).cpu().numpy().astype('uint8').transpose(1, 2, 0)
                Image.fromarray(log_lq_png).convert('L').save(os.path.join(exp_dir,   f"lq_{global_step:06d}.png"))
                Image.fromarray(log_gt_png).convert('L').save(os.path.join(exp_dir,   f"gt_{global_step:06d}.png"))
                Image.fromarray(log_pred_png).convert('L').save(os.path.join(exp_dir, f"out_{global_step:06d}.png"))

            # Compute full-reference metrics
            psnr, ssim, lpips_val = compute_full_reference_metrics( log_gt.to(device).unsqueeze(0), log_pred.unsqueeze(0) )
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_val)
            # print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, E*: {lpips_val:.4f}")
            global_step += 1

        print(f"PSNR: {np.mean(psnr_list):.2f} dB, SSIM: {np.mean(ssim_list):.4f}, lpips: {np.mean(lpips_list):.4f}")
        with open(os.path.join(exp_dir, "metrics.txt"), "a") as f:
            f.write(f"Frame: {ididid} PSNR: {np.mean(psnr_list):.4f} dB, SSIM: {np.mean(ssim_list):.4f}, lpips: {np.mean(lpips_list):.4f}\n")
        total_lpips += np.mean(lpips_list)
        total_psnr += np.mean(psnr_list)
        total_ssim += np.mean(ssim_list)
        psnr_list = []
        ssim_list = []
        lpips_list = []
        total_reconstructed_frames += 1

    with open(os.path.join(exp_dir, "cumulative_metrics.txt"), "w") as f:
        f.write(f"PSNR: {total_psnr / total_reconstructed_frames:.4f} dB, \
                    SSIM: {total_ssim / total_reconstructed_frames:.4f}, \
                    lpips: {total_lpips / total_reconstructed_frames:.4f}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)