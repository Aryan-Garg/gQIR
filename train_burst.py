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
from tqdm import tqdm
import lpips

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

# Debugging libs: ###############
# import matplotlib.pyplot as plt
# import numpy as np
#################################

from gqvr.utils.common import instantiate_from_config, calculate_psnr_pt, to
from gqvr.model.vae import AutoencoderKL

from gqvr.model.core_raft.raft import RAFT
from gqvr.model.core_raft.utils import flow_viz
from gqvr.model.core_raft.utils.utils import InputPadder

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

    warped = F.grid_sample(x, new_grid, align_corners=True)
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


def compute_burst_loss(gt, xhat_lq, lpips_model, scales, loss_mode="gt_perceptual"):
    #  "mse_ls", "ls_only", "ls_gt", "ls_gt_perceptual"
    mse_loss = 0.
    # ls_loss = 0.
    gt_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"mse": mse_loss, "perceptual": perceptual_loss, "l1_loss": gt_loss}
    if "mse" in loss_mode:
        mse_loss = scales.mse * F.mse_loss(xhat_lq, gt, reduction="sum")
        loss_dict["mse"] = mse_loss.item()
    elif "gt" in loss_mode:
        gt_loss = scales.l1 * F.l1_loss(xhat_lq, gt, reduction="sum")
        loss_dict["l1_loss"] = gt_loss.item()

    if "perceptual" in loss_mode:
        perceptual_loss = scales.perceptual * lpips_model(xhat_lq, gt)
        loss_dict["perceptual"] = perceptual_loss.item()

    total_loss = mse_loss + gt_loss + perceptual_loss
    return total_loss, loss_dict


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
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
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

    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model_path, subfolder="text_encoder", dtype=weight_dtype).to(device)
    text_encoder.eval().requires_grad_(False)

    def encode_prompt(prompt):
        txt_ids = tokenizer(
            prompt,
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
    ls_burst_unet.eval().requires_grad_(False)

    target_modules = cfg.lora_modules
    G_lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    ls_burst_unet.add_adapter(G_lora_cfg)
    lora_params = list(filter(lambda p: p.requires_grad, ls_burst_unet.parameters()))
    assert lora_params, "Failed to find lora parameters"
    for p in lora_params:
        p.data = p.to(torch.float32)

    # print(f"\n[~] All trainable RAFT model parameters: {(sum(p.numel() for p in raft_model.parameters() if p.requires_grad)) / 1e6:.2f}M")
    print(f"\n[~] All RAFT model parameters: {(sum(p.numel() for p in raft_model.parameters())) / 1e6:.2f}M")
    # print(f"[~] All trainable VAE model parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"[~] All VAE model parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")
    print(f"[~] All trainable burst Layer parameters: {sum(p.numel() for p in ls_burst_unet.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"[~] All burst Layer parameters: {sum(p.numel() for p in ls_burst_unet.parameters()) / 1e6:.2f}M")
    print(f"[~] All text encoder parameters: {(sum(p.numel() for p in text_encoder.parameters())) / 1e6:.2f}M\n")

    
    # Setup optimizer:
    opt = torch.optim.AdamW(ls_burst_unet.parameters(), 
        lr=cfg.lr_burst_model, 
        **cfg.opt_kwargs)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.train.batch_size,
        num_workers=cfg.dataset.train.num_workers,
        drop_last=True,
    )

    batch_transform = instantiate_from_config(cfg.dataset.batch_transform)

    raft_model, opt, loader = accelerator.prepare(
        raft_model, opt, loader
    )
    raft_model = accelerator.unwrap_model(raft_model) 

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.max_train_steps
    step_l1_loss = []
    step_perceptual_loss = []
    step_flow_loss = []
    epoch = 0
    epoch_loss = []

   
    with warnings.catch_warnings():
        # avoid warnings from lpips internal
        warnings.simplefilter("ignore")
        lpips_model = (
            lpips.LPIPS(net="vgg", verbose=accelerator.is_local_main_process)
            .eval()
            .to(device)
        )

    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # Training loop:
    pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_local_main_process,
            unit="step",
            total=max_steps
        )
    while global_step < max_steps:
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
        
            lqs = batch["lqs"].permute(0, 1, 4, 2, 3) # B T H W C
            gts = batch["gts"].permute(0, 1, 4, 2, 3) # B T C H W to match decoder output
            center_gt = gts[:, gts.size(1) // 2, ...] # B 3 H W
            del gts # save VRAM
            torch.cuda.empty_cache()

            reconstructed_lqs = []
            with torch.no_grad():
                for t in range(lqs.size(1)):
                    lq_t = lqs[:, t, ...] # B C H W
                    recon_lq_t = vae.decode(vae.encode(lq_t).mode())
                    reconstructed_lqs.append(recon_lq_t)
            
            reconstructed_lqs = torch.stack(reconstructed_lqs, dim=1) # B T C H W
            del lqs
            torch.cuda.empty_cache()

            # Merge lqs here using RAFT
            center_t = reconstructed_lqs.size(1) // 2 
            lq_center = reconstructed_lqs[:, center_t, ...] # B C H W

            flow_vectors = []
            for t in range(reconstructed_lqs.size(1)):
                lq_t = reconstructed_lqs[:, t, ...] # B C H W
                ls_in = lq_t.float()
                center_in = lq_center.float()
                if t < center_t:
                    _, flow_bw = raft_model(center_in, ls_in, iters=20, test_mode=True) # B 2 64 64
                else:
                    _, flow_bw = raft_model(ls_in, center_in, iters=20, test_mode=True) # B 2 64 64

                # Downsample flow_bw to match latent space size (1, 4, 64, 64) with the highest precision possible
                flow_bw = F.interpolate(flow_bw, size=(64, 64), mode='bilinear', align_corners=True)
                flow_vectors.append(flow_bw)

            latents = []
            for t in range(reconstructed_lqs.size(1)):
                latents.append(vae.encode(reconstructed_lqs[:, t, ...]).mode()) # B 4 64 64

            aligned_latents = []
            for t in range(reconstructed_lqs.size(1)):
                latent_t = latents[t] # B 4 64 64
                if t == center_t:
                    aligned_latents.append(latent_t)
                    continue
                flow_bw = flow_vectors[t] # B 2 64 64
                warped_latent = differentiable_warp(latent_t, flow_bw) # B 4 64 64
                aligned_latents.append(warped_latent)
            
            aligned_latents = torch.stack(aligned_latents, dim=1) # B T C H W
            # Average of aligned lqs 
            merged_latent = torch.mean(aligned_latents, dim=1, keepdim=False) # B C H W
            del aligned_latents
            torch.cuda.empty_cache()

            z_in = (merged_latent * 0.18215).to(weight_dtype) # vae scaling factor
            timesteps = torch.full((1,), cfg.model_t, dtype=torch.long, device=device)
            eps = ls_burst_unet(
                z_in,
                timesteps,
                encoder_hidden_states=encode_prompt("")["text_embed"],
            ).sample
            z = scheduler.step(eps, cfg.coeff_t, z_in).pred_original_sample
            decoded_refined = (vae.decode(z.float() / 0.18215)).float() # B 3 H W

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, loss_dict = compute_burst_loss(center_gt, decoded_refined, lpips_model, scales=cfg.loss_scales, loss_mode="gt_perceptual")
            
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            accelerator.wait_for_everyone()

            pbar.update(1)
            global_step += 1
            step_l1_loss.append(loss_dict["l1_loss"])
            step_perceptual_loss.append(loss_dict["perceptual"])
            epoch_loss.append(loss.item())

            # Log loss values:
            if global_step % cfg.log_every == 0 or global_step == 1:
                # Gather values from all processes
                avg_l1_loss = (
                    accelerator.gather(
                        torch.tensor(step_l1_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_perceptual_loss = (
                    accelerator.gather(
                        torch.tensor(step_perceptual_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                step_flow_loss.clear()
                step_l1_loss.clear()
                step_perceptual_loss.clear()

                if accelerator.is_local_main_process:
                    writer.add_scalar("train/l1_loss_step", avg_l1_loss, global_step)
                    writer.add_scalar("train/perceptual_loss_step", avg_perceptual_loss, global_step)

            # Save out & gt on disk
            if global_step % cfg.log_every == 0 or global_step == 1:
                flow_bw_furthest = flow_vectors[0] if center_t >= (reconstructed_lqs.size(1) // 2) else flow_vectors[-1]
                flow_img = flow_viz.flow_to_image(flow_bw_furthest[0].permute(1, 2, 0).cpu().numpy())
                Image.fromarray(flow_img).save(os.path.join(exp_dir, f"flow_furthest_{global_step:06d}.png"))
                pred = (((decoded_refined+1)/2.) * 255.).detach().cpu().numpy().astype('uint8')
                center_gt = (((center_gt+1)/2.) * 255.).cpu().numpy().astype('uint8')
                Image.fromarray(pred[0].transpose(1, 2, 0)).save(os.path.join(exp_dir, f"merged_burst_{global_step:06d}.png"))
                Image.fromarray(center_gt[0].transpose(1, 2, 0)).save(os.path.join(exp_dir, f"gt_center_{global_step:06d}.png"))

            # Save checkpoint:
            if global_step % cfg.checkpointing_steps == 0:
                if accelerator.is_local_main_process:
                    checkpoint = ls_burst_unet.state_dict()
                    ckpt_path = f"{ckpt_dir}/ls_burst_unet_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    

            if global_step == max_steps:
                break

        pbar.set_description(
            f"Epoch: {epoch:04d}, Global Step: {global_step:06d}, Loss: {loss.item():.6f}"
        )
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("train/total_loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("Done!")
        writer.close()
        pbar.close()

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)