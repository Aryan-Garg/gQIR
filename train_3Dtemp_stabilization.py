import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from argparse import ArgumentParser
import warnings

from omegaconf import OmegaConf
import torch
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
from gqvr.model.temporal_stabilizer import TemporalConsistencyLayer
from gqvr.model.core_raft.raft import RAFT
from gqvr.model.core_raft.utils import flow_viz
from gqvr.model.core_raft.utils.utils import InputPadder



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


def compute_stage3_loss(preds, gts, lpips_model, raft_model, loss_mode, scales):
    flow_loss = 0.
    l1_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"flow": flow_loss, "perceptual": perceptual_loss, "L1_loss": l1_loss}
    if "flow" in loss_mode:
        flow_loss = scales.flow * compute_flow_loss(preds, gts, raft_model)
        loss_dict["flow"] = flow_loss.item()

        if "gt" in loss_mode:
            gt_loss = scales.gt * F.l1_loss(preds, gts, reduction="sum")
            loss_dict["L1_loss"] = gt_loss.item()

        if "perceptual" in loss_mode:
            perceptual_loss = 0.
            for i in range(len(gts.size(1))):
                perceptual_loss = perceptual_loss + (scales.perceptual_gt * lpips_model(preds[:, i, :, :, :], gts[:, i, :, :, :])).item()
            loss_dict["perceptual"] = perceptual_loss
    else:
        raise NotImplementedError("[!] Always use Optical Flow Warping Loss for stage 3")

    total_loss = flow_loss + gt_loss + perceptual_loss
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

    vae.requires_grad_(False)

    temp_stabilizer = TemporalConsistencyLayer().to(device)
    temp_stabilizer.train().requires_grad_(True)

    vae.post_quant_conv.to(device)
    vae.decoder.to(device)
    # def decode_latent(z):
    #     z = z.to(post_quant_conv.weight.dtype)
    #     z = post_quant_conv(z)
    #     dec = decoder(z)
    #     return dec.float()

    class RAFT_args:
        mixed_precision = False
        small = False
        alternate_corr = False
        dropout = False
    raft_args = RAFT_args()

    raft_model = RAFT(raft_args)
    raft_things_dict = torch.load("/home/argar/apgi/gQVR/pretrained_checkpoints/raft_models_weights/raft-things.pth")
    corrected_state_dict = {}
    for k, v in raft_things_dict.items():
        k2 = ".".join(k.split(".")[1:])
        corrected_state_dict[k2] = v
    raft_model.load_state_dict(corrected_state_dict)
    raft_model.eval().requires_grad_(False).to(device)

    # Setup optimizer:
    opt = torch.optim.AdamW(temp_stabilizer.parameters(), 
        lr=cfg.lr_temp_stabilizer, 
        **cfg.opt_kwargs)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.train.batch_size,
        num_workers=cfg.dataset.train.dataloader_num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.dataset.val.batch_size,
        num_workers=cfg.dataset.val.num_workers,
        shuffle=False,
        drop_last=False,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset)} videos from {dataset.file_list}")

    batch_transform = instantiate_from_config(cfg.dataset.batch_transform)

    # Prepare models for training/inference:
    vae.to(device)
    temp_stabilizer, opt, loader, val_loader = accelerator.prepare(
        temp_stabilizer, opt, loader, val_loader
    )
    temp_stabilizer = accelerator.unwrap_model(temp_stabilizer)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.max_train_steps
    step_loss = []
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
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_local_main_process,
            unit="batch",
            total=len(loader),
        )

        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            if cfg.dataset.train.precomputed_latents:
                zs, gts = batch # B T 4 64 64, B T H W C
                zs.to(device)
                stable_zs = temp_stabilizer(zs) # B T 4 64 64
                preds = []
                for i in range(len(stable_zs.size(1))): # at each time step
                    pred = vae.decode(stable_zs[:, i, :, :, :])
                    preds.append(pred.permute(0, 2, 3, 1)) # B C H W -> B H W C
                preds.stack(preds, dim=1) # B T H W C (same as gts!)
                loss, loss_dict = compute_stage3_loss(preds, gts)
            else:
                raise NotImplementedError("[!] Precomputing latents saves tons of VRAM! Don't train without pre-computing unless you are FAANG")

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_flow_loss.append(loss_dict["flow_loss"])
            step_l1_loss.append(loss_dict["l1_loss"])
            step_perceptual_loss.append(loss_dict["perceptual"])
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.log_every == 0:
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
                step_loss.clear()
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
                    checkpoint = temp_stabilizer.state_dict()
                    ckpt_path = f"{ckpt_dir}/tempStab_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # Log images
            if global_step % cfg.log_image_steps == 0 or global_step == 1:
                temp_stabilizer.eval()
                N = 8
                log_gt, log_pred = gts[:, :N, ...], preds[:, :N, ...]
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/pred", log_pred),
                        ("image/gt", log_gt),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)
                temp_stabilizer.train()

            # Evaluate model:
            if global_step % cfg.val_every == 0:
                temp_stabilizer.eval() 
                # NOTE: eval() only halts BN stat accumulation & disables dropout. grad computation can still happen! Use with torch.no_grad() around model.

                val_loss = []
                val_lpips_loss = []
                val_psnr = []
                val_pbar = tqdm(
                    iterable=None,
                    disable=not accelerator.is_local_main_process,
                    unit="batch",
                    total=len(val_loader),
                    leave=False,
                    desc="Validation",
                )
                for val_batch in val_loader:
                    to(val_batch, device)
                    val_batch = batch_transform(val_batch)
                    val_zs, val_gts = val_batch # B T 4 64 64, B T H W C
                    with torch.no_grad():
                        stable_zs = temp_stabilizer(val_zs) # B T 4 64 64
                        val_preds = []
                        for i in range(len(stable_zs.size(1))): # at each time step
                            val_pred = vae.decode(stable_zs[:, i, :, :, :])
                            val_preds.append(val_pred.permute(0, 2, 3, 1)) # B C H W -> B H W C
                        val_preds.stack(preds, dim=1) # B T H W C (same as gts!)
                        vloss, vloss_dict = compute_stage3_loss(val_preds, val_gts)
                        psnr = 0.
                        for i in range(len(gts.size(1))):
                            psnr = psnr + calculate_psnr_pt(val_preds[:, i, ...], val_gts[:, i, ...], crop_border=0).mean().item()
                        val_psnr.append(psnr)
                        val_loss.append(vloss.item())
                        val_lpips_loss.append(vloss_dict['perceptual'])
                    val_pbar.update(1)

                val_pbar.close()
                avg_val_loss = (
                    accelerator.gather(
                        torch.tensor(val_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_lpips = (
                    accelerator.gather(
                        torch.tensor(val_lpips_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_psnr = (
                    accelerator.gather(
                        torch.tensor(val_psnr, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                if accelerator.is_local_main_process:
                    for tag, val in [
                        ("val/total_loss", avg_val_loss),
                        ("val/lpips", avg_val_lpips),
                        ("val/psnr", avg_val_psnr),
                    ]:
                        writer.add_scalar(tag, val, global_step)
                temp_stabilizer.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
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
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)