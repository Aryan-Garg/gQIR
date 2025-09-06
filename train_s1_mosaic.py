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


def compute_loss(gt, z_gt, z_pred, xhat_gt, xhat_lq, lpips_model, loss_mode, scales):
    #  "mse_ls", "ls_only", "ls_gt", "ls_gt_perceptual"
    mse_loss = 0.
    ls_loss = 0.
    gt_loss = 0.
    perceptual_loss = 0.
    loss_dict = {"mse": mse_loss, "lsa": ls_loss, "perceptual": perceptual_loss, "gt_loss": gt_loss}
    if "ls" in loss_mode:
        ls_loss = scales.lsa * F.mse_loss(z_pred, z_gt, reduction="mean")
        loss_dict["lsa"] = ls_loss.item()
        if "mse" in loss_mode:
            mse_loss = scales.mse * F.mse_loss(xhat_lq, xhat_gt, reduction="mean")
            loss_dict["mse"] = mse_loss.item()
        elif "gt" in loss_mode:
            gt_loss = scales.gt * F.l1_loss(xhat_gt, gt, reduction="mean")
            loss_dict["gt_loss"] = gt_loss.item()
        if "perceptual" in loss_mode:
            perceptual_loss = (scales.perceptual_gt * lpips_model(xhat_gt, gt)) + \
                (scales.perceptual_lq * lpips_model(xhat_lq, gt))
            loss_dict["perceptual"] = perceptual_loss.item()
    else:
        raise NotImplementedError("[!] Always use Latent Space Alignment (LSA) loss")

    total_loss = mse_loss + ls_loss + gt_loss + perceptual_loss
    return total_loss, loss_dict
    

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(310)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    assert cfg.train.loss_mode in ["mse_ls", "ls_only", "ls_gt", "ls_gt_perceptual"], f"Please choose a supported loss_mode from: ['mse_ls', 'ls_only', 'ls_gt', 'ls_gt_perceptual']"
    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    # Create & load VAE from pretrained SD model:
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    vae = AutoencoderKL(cfg.model.vae_cfg.ddconfig, cfg.model.vae_cfg.embed_dim)
    vae_sd = {k.replace("first_stage_model.", ""): v for k, v in sd.items() if k.startswith("first_stage_model.")}
    missing_keys, unexpected_keys = vae.load_state_dict(vae_sd, strict=False)
    if accelerator.is_main_process:
        print(
            f"Load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unexpected weights: {unexpected_keys}\n"
            f"missing weights: {missing_keys}"
        )
    
    # Make the encoder & quant_conv trainable and rest frozen
    for name, p in vae.named_parameters():
        p.requires_grad = True if "encoder" in name else False
        if "quant_conv" in name and not "post" in name:
            p.requires_grad = True 
        # print(f"{name} -> {p.shape} isTrainable? {p.requires_grad}")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        list(vae.encoder.parameters()) + list(vae.quant_conv.parameters()), 
        lr=cfg.train.learning_rate, 
        weight_decay=0)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training/inference:
    vae.to(device)
    vae.encoder, opt, loader, val_loader = accelerator.prepare(
        vae.encoder, opt, loader, val_loader
    )
    vae.encoder = accelerator.unwrap_model(vae.encoder)

    # for name, p in vae.named_parameters():
    #     print(f"{name} -> {p.shape} isTrainable? {p.requires_grad}")
    # print("Printed after accelerator prepares")
    # exit()

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    step_ls_loss = []
    step_gt_loss = []
    step_perceptual_loss = []
    epoch = 0
    epoch_loss = []

    if "perceptual" in cfg.train.loss_mode:
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
            gt, lq, prompt, gt_path = batch
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # Train step:
            gt_latent = vae.encode(gt).mode()
            pred_latent = vae.encode(lq).mode()
            xhat_lq = vae.decode(pred_latent)
            xhat_gt = vae.decode(gt_latent)
            loss, loss_dict = compute_loss(gt, gt_latent, pred_latent, xhat_gt, xhat_lq, 
                                           lpips_model, cfg.train.loss_mode, cfg.train.loss_scales)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss_dict["mse"])
            step_ls_loss.append(loss_dict["lsa"])
            step_gt_loss.append(loss_dict["gt_loss"])
            step_perceptual_loss.append(loss_dict["perceptual"])
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0:
                # Gather values from all processes
                avg_mse_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_ls_loss = (
                    accelerator.gather(
                        torch.tensor(step_ls_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_gt_loss = (
                    accelerator.gather(
                        torch.tensor(step_gt_loss, device=device).unsqueeze(0)
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
                step_ls_loss.clear()
                step_gt_loss.clear()
                step_perceptual_loss.clear()

                if accelerator.is_local_main_process:
                    writer.add_scalar("train/mse_loss_step", avg_mse_loss, global_step)
                    writer.add_scalar("train/ls_loss_step", avg_ls_loss, global_step)
                    writer.add_scalar("train/gt_loss_step", avg_gt_loss, global_step)
                    writer.add_scalar("train/perceptual_loss_step", avg_perceptual_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0:
                if accelerator.is_local_main_process:
                    checkpoint = vae.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # Log images
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                vae.encoder.eval()
                vae.quant_conv.eval()
                N = 12
                log_gt, log_lq = gt[:N], lq[:N]
                with torch.no_grad():
                    log_pred = vae.decode(vae.encode(log_lq).mode())
                    log_pred_gt = vae.decode(vae.encode(log_gt).mode())
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/pred_gt", (log_pred_gt+1) / 2),
                        ("image/pred", (log_pred+1) / 2),
                        ("image/gt", (log_gt + 1) / 2),
                        ("image/lq", (log_lq + 1) / 2),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)
                vae.encoder.train()
                vae.quant_conv.train()

            # Evaluate model:
            if global_step % cfg.train.val_every == 0:
                vae.encoder.eval() 
                # NOTE: eval() only halts BN stat accumulation & disables dropout. grad computation can still happen! Use with torch.no_grad() around model.
                vae.quant_conv.eval()
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
                    val_gt, val_lq, val_prompt, val_gt_path = val_batch
                    val_gt = (
                        rearrange(val_gt, "b h w c -> b c h w")
                        .contiguous()
                        .float()
                    )
                    val_lq = (
                        rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
                    )
                    with torch.no_grad():

                        lq_z = vae.encode(val_lq).mode()
                        gt_z = vae.encode(val_gt).mode()
                        xhat_lq = vae.decode(lq_z)
                        xhat_gt = vae.decode(gt_z)
                        vloss, vloss_dict = compute_loss(val_gt, gt_z, lq_z, xhat_gt, xhat_lq, 
                                           lpips_model, cfg.train.loss_mode, cfg.train.loss_scales)

                        val_psnr.append(
                            calculate_psnr_pt(xhat_lq, val_gt, crop_border=0)
                            .mean()
                            .item()
                        )
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
                        ("val/loss", avg_val_loss),
                        ("val/lpips", avg_val_lpips),
                        ("val/psnr", avg_val_psnr),
                    ]:
                        writer.add_scalar(tag, val, global_step)
                vae.encoder.train()
                vae.quant_conv.train()

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
            writer.add_scalar("train/loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)