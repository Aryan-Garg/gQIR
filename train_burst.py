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
from diffusers import StableDiffusionPipeline
import lpips

# Debugging libs: ###############
# import matplotlib.pyplot as plt
# import numpy as np
#################################

from gqvr.utils.common import instantiate_from_config, calculate_psnr_pt, to
from gqvr.model.vae import AutoencoderKL

from gqvr.model.core_raft.raft import RAFT
from gqvr.model.core_raft.utils import flow_viz
from gqvr.model.core_raft.utils.utils import InputPadder

from gqvr.model.generator import SD2Enhancer

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
    loss_dict = {"mse": mse_loss, "perceptual": perceptual_loss, "gt_loss": gt_loss}
    if "mse" in loss_mode:
        mse_loss = scales.mse * F.mse_loss(xhat_lq, gt, reduction="sum")
        loss_dict["mse"] = mse_loss.item()
    elif "gt" in loss_mode:
        gt_loss = scales.l1 * F.l1_loss(xhat_lq, gt, reduction="sum")
        loss_dict["gt_loss"] = gt_loss.item()

    if "perceptual" in loss_mode:
        perceptual_loss = scales.perceptual * lpips_model(xhat_lq, gt)
        loss_dict["perceptual"] = perceptual_loss.item()

    total_loss = mse_loss + ls_loss + gt_loss + perceptual_loss
    return total_loss, loss_dict


def compute_stage3_loss(preds, gts, lpips_model, raft_model, loss_mode, scales):
    flow_loss = 0.
    l1_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"flow": flow_loss, "perceptual": perceptual_loss, "L1_loss": l1_loss}
    if "flow" in loss_mode:
        flow_loss = scales.flow * compute_flow_loss(preds, gts, raft_model)
        loss_dict["flow"] = flow_loss.item()

        if "gt" in loss_mode:
            l1_loss = scales.l1 * F.l1_loss(preds, gts, reduction="sum")
            loss_dict["L1_loss"] = l1_loss.item()

        if "perceptual" in loss_mode:
            perceptual_loss = 0.
            for i in range(gts.size(1)):
                perceptual_loss = perceptual_loss + (scales.perceptual * lpips_model(preds[:, i, :, :, :], gts[:, i, :, :, :])).item()
            loss_dict["perceptual"] = perceptual_loss
    else:
        raise NotImplementedError("[!] Always use Optical Flow Warping Loss for stage 3")

    total_loss = flow_loss + l1_loss + perceptual_loss
    return total_loss, loss_dict


def compute_stage3_loss_streaming(zs, gts, vae, lpips_model, raft_model, loss_mode, scales, chunk_T=1):
    B, T, Cz, Hz, Wz = zs.shape
    _, _, Cx, Hx, Wx = gts.shape
    do_flow = ("flow"       in loss_mode)
    do_l1   = ("l1"         in loss_mode) 
    do_lpips= ("perceptual" in loss_mode)
    total_flow = torch.zeros([], device=zs.device)
    total_l1   = torch.zeros([], device=zs.device)
    total_lp   = torch.zeros([], device=zs.device)

    prev_dec = None  # decoded pred at t-1 (for flow)

    for t0 in range(0, T, chunk_T):
        t1 = min(T, t0 + chunk_T)
        z_chunk = zs[:, t0:t1, ...].to(torch.float32)          # [1, tc, 4, 64, 64]; fp32 for best optical flow computations
        gt_chunk = gts[:, t0:t1, ...].to(torch.float32)         # [1, tc, 3, H, W]

        # treat time as batch; keep autocast for speed/VRAM
        tc = t1 - t0
        z_flat = z_chunk.reshape(B*tc, Cz, Hz, Wz)

        # Decode
        dec_flat = vae.decode(z_flat)  # [B*tc,3,H,W], fp32 return
        dec_chunk = dec_flat.reshape(B, tc, 3, Hx, Wx)            # [1, tc, 3, H, W]

        # Accumulate per-frame losses immediately; free decoded frames ASAP
        for i in range(tc):
            t = t0 + i
            dec_t = dec_chunk[:, i, ...]  # [1,3,H,W]
            gt_t  = gt_chunk[:, i, ...]   # [1,3,H,W]

            # Perceptual & L1
            if do_l1:
                total_l1 = total_l1 + scales.l1 * F.l1_loss(dec_t, gt_t)

            if do_lpips:
                # LPIPS expects [-1,1] often; adjust if your gts/dec are [0,1].
                # If both already in [0,1], many codebases still feed as-is.
                total_lp = total_lp + scales.perceptual * lpips_model(dec_t, gt_t)

            # Flow (use decoded preds; change to GT flow supervision if you prefer)
            if do_flow and prev_dec is not None:
                _, flow_fw = raft_model(prev_dec, dec_t, iters=20, test_mode=True)
                _, flow_bw = raft_model(dec_t, prev_dec, iters=20, test_mode=True)
                occ, warp_to_prev = detect_occlusion(flow_fw, flow_bw, dec_t)
                noc = 1.0 - occ
                diff = (prev_dec - warp_to_prev) * noc
                # robust average
                denom = torch.clamp(noc.sum(), min=1.0)
                total_flow = total_flow + scales.flow * (diff.pow(2).sum() / denom)

            prev_dec = dec_t.detach()  # keep the decoded prior for flow next step; detach to avoid long graph

        # Free chunk tensors ASAP
        del z_flat, dec_flat, dec_chunk, z_chunk, gt_chunk
        # torch.cuda.empty_cache()

    total_loss = total_flow + total_l1 + total_lp
    loss_dict = {
        "flow_loss": float(total_flow.detach().item()),
        "l1_loss": float(total_l1.detach().item()),
        "perceptual": float(total_lp.detach().item()),
    }
    return total_loss, loss_dict


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
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

    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    # vae.post_quant_conv.to(device)
    # vae.decoder.to(device)
    # def decode_latent(z):
    #     z = z.to(post_quant_conv.weight.dtype)
    #     z = post_quant_conv(z)
    #     dec = decoder(z)
    #     return dec.float()

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
    raft_model.train().requires_grad_(True).to(device)

    if cfg.with_unet:
        sd2_enhancer = SD2Enhancer(
            base_model_path     =   cfg.base_model_path,
            weight_path         =   cfg.weight_path,
            lora_modules        =   cfg.lora_modules,
            lora_rank           =   cfg.lora_rank,
            model_t             =   cfg.model_t,
            coeff_t             =   cfg.coeff_t,
            vae_cfg             =   cfg.model.vae_cfg,
            device              =   args.device, 
        )
        sd2_enhancer.init_models(init_vae = False)

    # Setup optimizer:
    opt = torch.optim.AdamW(raft_model.parameters(), 
        lr=cfg.lr_raft_model, 
        **cfg.opt_kwargs)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.train.batch_size,
        num_workers=cfg.dataset.train.num_workers,
        drop_last=True,
    )
    # val_dataset = instantiate_from_config(cfg.dataset.val)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=cfg.dataset.val.batch_size,
    #     num_workers=cfg.dataset.val.num_workers,
    #     drop_last=True,
    # )

    batch_transform = instantiate_from_config(cfg.dataset.batch_transform)

    # Prepare models for training/inference:
    # temp_stabilizer, opt, loader, val_loader = accelerator.prepare(
    #     temp_stabilizer, opt, loader, val_loader
    # )

    raft_model.to(device)
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

            with torch.no_grad():
                vae_latents = []
                for t in range(lqs.size(1)):
                    lq_t = lqs[:, t, ...] # B C H W
                    latent_t = vae.encode(lq_t).mode() # B 4 64 64
                    vae_latents.append(latent_t)
                zs = torch.stack(vae_latents, dim=1) # B T 4 64 64
                # print("[+] All latents from lq video stacked")
                # Free up VRAM
                del vae_latents, lqs
                torch.cuda.empty_cache()
                # Warp all latents to center latent using raft
                aligned_zs = []
                center_t = zs.size(1) // 2
                z_center = zs[:, center_t, ...] # B 4 64 64
                for t in range(zs.size(1)):
                    if t == center_t:
                        aligned_zs.append(z_center)
                        continue
                    z_t = zs[:, t, ...] # B 4 64 64
                    ls_in = z_t[:, 1, ...].repeat(1,3,1,1).float()
                    center_in = z_center[:, 1, ...].repeat(1,3,1,1).float()
                    ls_in_up = F.interpolate(ls_in, size=(256,256), mode='bilinear', align_corners=False)
                    center_in_up = F.interpolate(center_in, size=(256,256), mode='bilinear', align_corners=False)
                    # print(f"[+] Computing flow between frames {t} and {center_t}\nls_in: {ls_in_up.shape}, center_in: {center_in_up.shape}")
                    if t < center_t:
                        # _, flow_fw = raft_model(ls_in_up, center_in_up, iters=20, test_mode=True) # B 2 64 64 --- NEed this only for flow loss
                        _, flow_bw = raft_model(center_in_up, ls_in_up, iters=20, test_mode=True) # B 2 64 64
                    else:
                        # _, flow_fw = raft_model(center_in_up, ls_in_up, iters=20, test_mode=True) # B 2 64 64
                        _, flow_bw = raft_model(ls_in_up, center_in_up, iters=20, test_mode=True) # B 2 64 64
                    
                    z_t_warped_c1 = differentiable_warp(F.interpolate(z_center[:, 0, ...].repeat(1,3,1,1).float(), 
                                                                        size=(256,256), mode='bilinear', align_corners=False), 
                                                        flow_bw)
                    z_t_warped_c2 = differentiable_warp(center_in_up, flow_bw)
                    z_t_warped_c3 = differentiable_warp(F.interpolate(z_center[:, 2, ...].repeat(1,3,1,1).float(), 
                                                                        size=(256,256), mode='bilinear', align_corners=False), 
                                                        flow_bw)
                    z_t_warped_c4 = differentiable_warp(F.interpolate(z_center[:, 3, ...].repeat(1,3,1,1).float(), 
                                                                        size=(256,256), mode='bilinear', align_corners=False),
                                                        flow_bw)
                    z_t_warped = torch.stack([z_t_warped_c1[:, 0, ...], 
                                            z_t_warped_c2[:, 0, ...], 
                                            z_t_warped_c3[:, 0, ...], 
                                            z_t_warped_c4[:, 0, ...]], dim=1)
                    z_t_warped = F.interpolate(z_t_warped, size=(64,64), mode='bilinear', align_corners=False)
                    aligned_zs.append(z_t_warped)
                # print("[+] All latents aligned to center frame")
                
                # Merge all aligned latents - weighted sum (further away frames have smaller alpha, since more erroneous flow)
                sum_merged_frame = torch.zeros_like(z_center)
                for t in range(zs.size(1)):
                    # Alpha is inversely proportionate to distance from center. All alphas sum to 1
                    alpha = (center_t - abs(t - center_t)) / zs.size(1)
                    sum_merged_frame = sum_merged_frame + alpha * aligned_zs[t]
                # print("[+] All latents merged to single latent")

                if cfg.with_unet: 
                    burst_latent = sd2_enhancer.forward_generator(sum_merged_frame.half()) # B 4 64 64
                else:
                    burst_latent = sum_merged_frame
                decoded = vae.decode(burst_latent) # B 3 H W
                # print("[+] Merged latent decoded to image") 
                
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, loss_dict = compute_burst_loss(center_gt, decoded, lpips_model, scales=cfg.loss_scales, loss_mode="gt_perceptual")
            

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            accelerator.wait_for_everyone()

            pbar.update(1)
            global_step += 1
            step_flow_loss.append(loss_dict["flow_loss"])
            step_l1_loss.append(loss_dict["l1_loss"])
            step_perceptual_loss.append(loss_dict["perceptual"])
            epoch_loss.append(loss.item())

            # Log loss values:
            if global_step % cfg.log_every == 0 or global_step == 1:
                # Gather values from all processes
                avg_flow_loss = (
                    accelerator.gather(
                        torch.tensor(step_flow_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
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
                    writer.add_scalar("train/flow_loss_step", avg_flow_loss, global_step)
                    writer.add_scalar("train/l1_loss_step", avg_l1_loss, global_step)
                    writer.add_scalar("train/perceptual_loss_step", avg_perceptual_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.checkpointing_steps == 0:
                if accelerator.is_local_main_process:
                    checkpoint = raft_model.state_dict()
                    ckpt_path = f"{ckpt_dir}/raft_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # Log images
            if global_step % cfg.log_image_steps == 0 or global_step == 1:
                N = min(8, gts.size(1))
                log_gt = (gts[:, :N, ...].cpu() + 1) / 2
                if accelerator.is_local_main_process: 
                    with torch.no_grad():
                        log_zs = stable_zs[:, :N]
                        log_preds = vae.decode(log_zs.squeeze(0))
                        log_preds = (log_preds.unsqueeze(0).cpu() + 1) / 2
                    
                    log_gt_arr = (log_gt * 255.).squeeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).contiguous().numpy()
                    log_pred_arr = (log_preds * 255.).squeeze(0).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).contiguous().numpy()

                    save_dir = os.path.join(
                        cfg.output_dir, "log_videos", f"{global_step:06}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    for i in range(N):
                        Image.fromarray(log_gt_arr[i]).save(os.path.join(save_dir, f"gt_frame_{i}.png"))
                        Image.fromarray(log_pred_arr[i]).save(os.path.join(save_dir, f"pred_frame_{i}.png"))

            # Evaluate model:
            # if global_step % cfg.val_every == 0 or global_step == 1:
            #     temp_stabilizer.eval() 
            #     # NOTE: eval() only halts BN stat accumulation & disables dropout. grad computation can still happen! Use with torch.no_grad() around model.

            #     val_loss = []
            #     val_lpips_loss = []
            #     val_psnr = []
            #     val_pbar = tqdm(
            #         iterable=None,
            #         disable=not accelerator.is_local_main_process,
            #         unit="batch",
            #         leave=False,
            #         desc="Validation",
            #     )
            #     for val_batch in val_loader:
            #         to(val_batch, device)
            #         val_batch = batch_transform(val_batch)
            #         val_zs = val_batch["latents"] # B T 4 64 64
            #         val_gts = val_batch["gts"]
            #         with torch.no_grad():
            #             stable_zs = temp_stabilizer(val_zs) # B T 4 64 64
            #             val_pred = vae.decode(stable_zs.squeeze(0))
            #             val_pred = val_pred.unsqueeze(0)
            #             with torch.amp.autocast("cuda", dtype=torch.float16):
            #                 vloss, vloss_dict = compute_stage3_loss_streaming(stable_zs, val_gts, vae,
            #                                                                   lpips_model, raft_model, 
            #                                                                   "l1_flow_perceptual", cfg.loss_scales,
            #                                                                   chunk_T=1)
            #             psnr = 0.
            #             for i in range(val_gts.size(1)):
            #                 psnr = psnr + calculate_psnr_pt(val_pred[:, i, ...], val_gts[:, i, ...], crop_border=0).mean().item()
            #             val_psnr.append(psnr)
            #             val_loss.append(vloss.item())
            #             val_lpips_loss.append(vloss_dict['perceptual'])
            #         val_pbar.update(1)

            #     val_pbar.close()
            #     avg_val_loss = (
            #         accelerator.gather(
            #             torch.tensor(val_loss, device=device).unsqueeze(0)
            #         )
            #         .mean()
            #         .item()
            #     )
            #     avg_val_lpips = (
            #         accelerator.gather(
            #             torch.tensor(val_lpips_loss, device=device).unsqueeze(0)
            #         )
            #         .mean()
            #         .item()
            #     )
            #     avg_val_psnr = (
            #         accelerator.gather(
            #             torch.tensor(val_psnr, device=device).unsqueeze(0)
            #         )
            #         .mean()
            #         .item()
            #     )
            #     if accelerator.is_local_main_process:
            #         for tag, val in [
            #             ("val/total_loss", avg_val_loss),
            #             ("val/lpips", avg_val_lpips),
            #             ("val/psnr", avg_val_psnr),
            #         ]:
            #             writer.add_scalar(tag, val, global_step)
            #     temp_stabilizer.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

            pbar.update(1)
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