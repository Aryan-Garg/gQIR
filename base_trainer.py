
import logging
import os
import shutil
from pathlib import Path
from typing import overload, List, Dict
import importlib
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.serialization import get_unsafe_globals_in_checkpoint, add_safe_globals
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor, AutoModelForImageTextToText
import lpips
import diffusers
from diffusers import AutoencoderKL
from PIL import Image
import math
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

from gqvr.model.discriminator import ImageConvNextDiscriminator
from gqvr.model.vae import AutoencoderKL
from gqvr.utils.common import instantiate_from_config, log_txt_as_img, print_vram_state, SuppressLogging
from gqvr.utils.ema import EMAModel
from gqvr.utils.tabulate import tabulate

from gqvr.model.core_raft.raft import RAFT
from gqvr.model.core_raft.utils import flow_viz
from gqvr.model.core_raft.utils.utils import InputPadder

from gqvr.model.temporal_stabilizer import TemporalConsistencyLayer

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class BatchInput:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError(f"Duplicated key in BatchInput: {name}")
        self.__dict__[name] = value

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value


############################## For internVL3-8B ##########################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        self.init_environment()
        self.init_models()
        self.summary_models()
        self.init_optimizers()
        self.init_dataset()
        self.prepare_all()

    def init_environment(self):
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
            # mixed_precision=self.config.mixed_precision,
            split_batches=True
        )
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_main_process:
            accelerator.init_trackers("train")
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model

    def init_models(self):
        if self.config.lifting_training:
            self.init_scheduler()
            self.init_text_models()
            self.init_vae()
            self.init_generator()
            self.init_lpips()
            self.init_temp_stabilizer()
            self.init_flow_model()
        else:
            self.init_scheduler()
            self.init_text_models()
            self.init_vae()
            self.init_generator()
            self.init_discriminator()
            self.init_lpips()

        if self.config.prompt_training:
            self.init_internVL() 
        
    def init_temp_stabilizer(self):
        self.temp_stabilizer = TemporalConsistencyLayer().to(self.device)
        self.temp_stabilizer.train().requires_grad_(True)

    def init_flow_model(self):
        class RAFT_args:
            mixed_precision = False
            small = False
            alternate_corr = False
            dropout = False
        raft_args = RAFT_args()

        self.raft_model = RAFT(raft_args)
        raft_things_dict = torch.load("/home/argar/apgi/gQVR/pretrained_checkpoints/raft_models_weights/raft-things.pth")
        corrected_state_dict = {}
        for k, v in raft_things_dict.items():
            k2 = ".".join(k.split(".")[1:])
            corrected_state_dict[k2] = v
        self.raft_model.load_state_dict(corrected_state_dict)
        self.raft_model.eval().requires_grad_(False).to(self.device)
        
    def init_internVL(self):
        torch_device = "cuda:7"
        path = 'OpenGVLab/InternVL3-8B'
        self.internVL3 = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            # device_map=device_map,
            token="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").eval().requires_grad_(False).to(torch_device)
        self.internVL3_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    @overload
    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        ...

    def init_vae(self):
        self.vae = AutoencoderKL(self.config.model.vae_cfg.ddconfig, self.config.model.vae_cfg.embed_dim)       
        # Load quanta VAE
        daVAE = torch.load(self.config.qvae_path, map_location="cpu")
        init_vae = {}
        vae_used = set()
        scratch_vae = self.vae.state_dict()
        for key in scratch_vae:
            if key not in daVAE:
                print(f"[!] {key} missing in daVAE")
                continue
            # print(f"Found {key} in daVAE. Loading...")
            init_vae[key] = daVAE[key].clone()
            vae_used.add(key)
        self.vae.load_state_dict(init_vae, strict=True)
        self.vae.to(device=self.device)
        vae_unused = set(daVAE.keys()) - vae_used
        
        if len(vae_unused) == 0:
            print(f"Loaded qVAE successfully")
        else:
            print(f"[!] VAE keys NOT used: {vae_unused}")

        self.vae.eval().requires_grad_(False)

    def init_lpips(self):
        with warnings.catch_warnings():
            # Suppress warnings from lpips
            warnings.simplefilter("ignore")
            self.net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.net_lpips.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    def init_discriminator(self):
        # Suppress logs from open-clip
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = ImageConvNextDiscriminator().to(device=self.device)
        self.D.train().requires_grad_(True)

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        logger.info(f"Model Summary:\n{table}")

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        if self.config.lifting_training:
            trainable_params = sum(p.numel() for p in self.temp_stabilizer.parameters() if p.requires_grad)
            print(f"\n[+] Optimizing {trainable_params} parameters for temporal stabilization\n")
            self.temp_stabilizer_params = list(filter(lambda p: p.requires_grad, self.temp_stabilizer.parameters()))
            self.temp_stabilizer_opt = optimizer_cls(
                self.temp_stabilizer_params,
                lr=self.config.lr_temp_stabilizer,
                **self.config.opt_kwargs,
            )
        else:
            self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
            self.G_opt = optimizer_cls(
                self.G_params,
                lr=self.config.lr_G,
                **self.config.opt_kwargs,
            )

            self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
            self.D_opt = optimizer_cls(
                self.D_params,
                lr=self.config.lr_D,
                **self.config.opt_kwargs,
            )

    def init_dataset(self):
        data_cfg = self.config.dataset
        dataset = instantiate_from_config(data_cfg.train)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=data_cfg.train.batch_size,
            num_workers=data_cfg.train.dataloader_num_workers,
        )
        self.batch_transform = instantiate_from_config(data_cfg.train.batch_transform)

    def prepare_all(self):
        if self.config.lifting_training:
            logger.info("[+] Preparing models, optimizers and dataloaders for image prior lifting")
            attrs = ["temp_stabilizer", "temp_stabilizer_opt", "dataloader"]
            prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
            for attr, obj in zip(attrs, prepared_objs):
                setattr(self, attr, obj)
        else:
            logger.info("Wrapping models, optimizers and dataloaders")
            attrs = ["G", "D", "G_opt", "D_opt", "dataloader"]
            prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
            for attr, obj in zip(attrs, prepared_objs):
                setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def force_optimizer_ckpt_safe(self, checkpoint_dir):
        def get_symbol(s):
            module_name, symbol_name = s.rsplit('.', 1)
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            return symbol

        for file_name in os.listdir(checkpoint_dir):
            if "optimizer" in file_name and not file_name.endswith("safetensors"):
                path = os.path.join(checkpoint_dir, file_name)
                unsafe_globals = get_unsafe_globals_in_checkpoint(path)
                logger.info(f"Unsafe globals in {path}: {unsafe_globals}")
                unsafe_globals = list(map(get_symbol, unsafe_globals))
                add_safe_globals(unsafe_globals)

    def attach_accelerator_hooks(self):
        ...

    def on_training_start(self):
        if self.config.lifting_training:
            # Build ema state dict
            logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
            if self.config.resume_from_checkpoint is not None and self.config.resume_ema:
                ema_resume_pth = os.path.join(self.config.resume_from_checkpoint, "ema_state_dict.pth")
            else:
                ema_resume_pth = None
            self.ema_handler = EMAModel(
                self.unwrap_model(self.temp_stabilizer),
                decay=self.config.ema_decay,
                use_ema=self.config.use_ema,
                ema_resume_pth=ema_resume_pth,
                verbose=self.accelerator.is_local_main_process,
            )
            global_step = 0
            if self.config.resume_from_checkpoint:
                path = self.config.resume_from_checkpoint
                ckpt_name = os.path.basename(path)
                logger.info(f"Resuming from checkpoint {path}")
                self.force_optimizer_ckpt_safe(path)
                self.accelerator.load_state(path)
                global_step = int(ckpt_name.split("-")[1])
                init_global_step = global_step
            else:
                init_global_step = 0
            self.global_step = global_step
            self.pbar = tqdm(
                range(0, self.config.max_train_steps),
                initial=init_global_step,
                desc="Steps",
                disable=not self.accelerator.is_main_process,
            )
        else:
            # Build ema state dict
            logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
            if self.config.resume_from_checkpoint is not None and self.config.resume_ema:
                ema_resume_pth = os.path.join(self.config.resume_from_checkpoint, "ema_state_dict.pth")
            else:
                ema_resume_pth = None
            self.ema_handler = EMAModel(
                self.unwrap_model(self.G),
                decay=self.config.ema_decay,
                use_ema=self.config.use_ema,
                ema_resume_pth=ema_resume_pth,
                verbose=self.accelerator.is_local_main_process,
            )

            global_step = 0
            if self.config.resume_from_checkpoint:
                path = self.config.resume_from_checkpoint
                ckpt_name = os.path.basename(path)
                logger.info(f"Resuming from checkpoint {path}")
                self.force_optimizer_ckpt_safe(path)
                self.accelerator.load_state(path)
                global_step = int(ckpt_name.split("-")[1])
                init_global_step = global_step
            else:
                init_global_step = 0

            self.global_step = global_step
            self.pbar = tqdm(
                range(0, self.config.max_train_steps),
                initial=init_global_step,
                desc="Steps",
                disable=not self.accelerator.is_main_process,
            )

    def get_internVL_prompt(self, gt_path):
        bs = len(gt_path)
        lst_prompts = []
        generation_config = dict(max_new_tokens=256, do_sample=True)
        for i in range(bs):
            pixel_values = load_image(gt_path[i], max_num=2).to(torch.bfloat16).to("cuda:7")
            # print(pixel_values.size())
            question = '<image>\nPlease describe the image in detail in a single paragraph. Explicitly write the legible text in the image.'
            response = self.internVL3.chat(self.internVL3_tokenizer, pixel_values, question, generation_config = generation_config)
            lst_prompts.append(response)
        return lst_prompts

    def prepare_batch_inputs(self, batch):
        if self.config.lifting_training:
            batch = self.batch_transform(batch)
            gt_vid, lq_vid, prompt, gt_path = batch

            gt_vid = gt_vid.permute(0, 1, 4, 2, 3) # N T C H W
            lq_vid = lq_vid.permute(0, 1, 4, 2, 3)
            if self.config.prompt_training:
                prompt = self.get_internVL_prompt(gt_path)
                # print(f"[+] InternVL3 Prompt: {prompt}")
            bs = len(prompt)
            c_txt = self.encode_prompt(prompt)

            z_lqs = []
            for i in range(lq_vid.size(1)):
                this_lq = lq_vid[:, i, :, :, :].to(self.weight_dtype)  # N C H W
                this_z_lq = self.vae.encode(this_lq).mode()
                z_lqs.append(this_z_lq.unsqueeze(1))  # N 1 C H W
            z_lq = torch.cat(z_lqs, dim=1)  # N T C H W

            timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
            self.batch_inputs = BatchInput(
                gt=gt_vid, 
                lq=lq_vid,
                z_lq=z_lq,
                c_txt=c_txt,
                timesteps=timesteps,
                prompt=prompt,
            )
        else:
            batch = self.batch_transform(batch)
            gt, lq, prompt, gt_path = batch
            # print(f"[+] gt_path: {gt_path}\n")
            gt = gt.permute(0, 3, 1, 2) # N C H W
            lq = lq.permute(0, 3, 1, 2)

            if self.config.prompt_training:
                prompt = self.get_internVL_prompt(gt_path)
                # print(f"[+] InternVL3 Prompt: {prompt}")

            bs = len(prompt)
            c_txt = self.encode_prompt(prompt)
            # NOTE:
            # .sample() => Adding some noise to the encoded latent --- for perception vs fidelity control
            # .mode() => For maximum fidelity 
            z_lq = self.vae.encode(lq.to(self.weight_dtype)).mode() 
            timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
            self.batch_inputs = BatchInput(
                gt=gt, 
                lq=lq,
                z_lq=z_lq,
                c_txt=c_txt,
                timesteps=timesteps,
                prompt=prompt,
            )

    @overload
    def forward_generator(self) -> torch.Tensor:
        ...

    @overload
    def collect_all_latents(self) -> torch.Tensor:
        ...

    @overload
    def forward_temp(self, z: torch.Tensor) -> torch.Tensor:
        ...
 
    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            self.unwrap_model(self.D).eval().requires_grad_(False)
            x = self.forward_generator()
            # print(f"Gen output: {x.shape}")
            self.G_pred = x
            loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
            loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
            loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
            loss_G = loss_l2 + loss_lpips + loss_disc
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()
            self.accelerator.wait_for_everyone()
            
        # Log something
        loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc)
        return loss_dict

    def optimize_discriminator(self):
        gt = self.batch_inputs.gt
        with torch.no_grad():
            x = self.forward_generator()
        self.G_pred = x
        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
            loss_D_fake, fake_logits = self.D(x, for_real=False, return_logits=True)
            loss_D = loss_D_real.mean() + loss_D_fake.mean()
        
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.D_params, self.config.max_grad_norm)
            self.D_opt.step()
            self.D_opt.zero_grad()
            self.accelerator.wait_for_everyone()
        
        loss_dict = dict(D=loss_D)
        # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
        return loss_dict

    def unload_vram(self):
        if hasattr(self, "vae"):
            self.vae.encoder.to("cpu")
            torch.cuda.empty_cache()
            logger.info("vae.encoder unloaded from VRAM")
        if hasattr(self, "G"):
            self.G.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Generator unloaded from VRAM")

    def reload_vram(self, vae_or_g="vae"):
        if vae_or_g == "vae":
            if hasattr(self, "vae"):
                self.vae.encoder.to(self.device)
                logger.info("vae.encoder reloaded to VRAM")
        else:
            if hasattr(self, "G"):
                self.G.to(self.device)
                logger.info("Generator reloaded to VRAM")

    def compute_flow_loss(self, pred_frames, gt_frames):
        B, T, C, H, W = pred_frames.shape
        err = 0.0 
        for i in range(T - 1):
            frame1 = pred_frames[:, i, ...]
            frame2 = gt_frames[:, i+1, ...]

            # Compute forward and backward flow using RAFT 
            _, flow_fw = self.raft_model(frame1, frame2, iters=20, test_mode=True) 
            _, flow_bw = self.raft_model(frame2, frame1, iters=20, test_mode=True) 

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

    def optimize_temp_stabilizer(self):
        with self.accelerator.accumulate(self.temp_stabilizer):
            z = self.collect_all_latents()
            z = z.to(self.weight_dtype)
            # ??? No need since ConvNext is no longer on VRAM
            # # Remove the UNet & Encoder of the VAE from the VRAM 
            self.unload_vram()
            self.temp_pred = self.forward_temp(z)
            # NOTE: collect_all_latents() and forward_temp() are separately defined to perform the unloading of models
            # since video training requires much more VRAM
            
            loss_l2 = F.mse_loss(self.temp_pred, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
            loss_lpips = self.net_lpips(self.temp_pred, self.batch_inputs.gt).mean() * self.config.lambda_lpips
            loss_flow = self.compute_flow_loss(self.temp_pred, self.batch_inputs.gt) * self.config.lambda_flow
            loss_temp_stabilizer = loss_l2 + loss_lpips + loss_flow

            self.accelerator.backward(loss_temp_stabilizer)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.temp_stabilizer_params, self.config.max_grad_norm)
            self.temp_stabilizer_opt.step()
            self.temp_stabilizer_opt.zero_grad()
            self.accelerator.wait_for_everyone()

        # Log something
        loss_dict = dict(temp_total=loss_temp_stabilizer, temp_mse=loss_l2, temp_lpips=loss_lpips, temp_flow=loss_flow)
        return loss_dict

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                bs = len(self.batch_inputs.lq)
                if self.config.lifting_training:
                    loss_dict = self.optimize_temp_stabilizer()

                    for k, v in loss_dict.items():
                        avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                        if k not in train_loss:
                            train_loss[k] = 0
                        train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps
                    
                    self.batch_count += 1
                    if self.accelerator.sync_gradients:
                        # update EMA
                        self.ema_handler.update()
                        _, _, peak = print_vram_state(None)
                        self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")
                    
                        self.global_step += 1
                        self.pbar.update(1)
                        log_dict = {}
                        for k in train_loss.keys():
                            log_dict[f"loss/{k}"] = train_loss[k]
                        train_loss = {}
                        self.accelerator.log(log_dict, step=self.global_step)
                        if self.global_step % self.config.log_image_steps == 0 or self.global_step == 1:
                            self.log_videos()
                        if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 10:
                            self.save_checkpoint()

                    if self.global_step >= self.config.max_train_steps:
                        break
                    
                    # ??? No need since ConvNext is no longer on VRAM
                    # # Reload VAE & UNet to VRAM
                    self.reload_vram("G")

                else:
                    generator_step = ((self.batch_count // self.config.gradient_accumulation_steps) % 2) == 0
                    if generator_step:
                        loss_dict = self.optimize_generator()
                    else:
                        loss_dict = self.optimize_discriminator()

                    for k, v in loss_dict.items():
                        avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                        if k not in train_loss:
                            train_loss[k] = 0
                        train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps

                    self.batch_count += 1
                    if self.accelerator.sync_gradients:
                        if generator_step:
                            # update EMA
                            self.ema_handler.update()
                        state = "Generator     Step" if not generator_step else "Discriminator Step"
                        _, _, peak = print_vram_state(None)
                        self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")

                    if self.accelerator.sync_gradients and not generator_step:
                        self.global_step += 1
                        self.pbar.update(1)
                        log_dict = {}
                        for k in train_loss.keys():
                            log_dict[f"loss/{k}"] = train_loss[k]
                        train_loss = {}
                        self.accelerator.log(log_dict, step=self.global_step)
                        if self.global_step % self.config.log_image_steps == 0 or self.global_step == 1:
                            self.log_images()
                        if self.global_step % self.config.log_grad_steps == 0 or self.global_step == 1:
                            self.log_grads()
                        if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 10:
                            self.save_checkpoint()

                    if self.global_step >= self.config.max_train_steps:
                        break
        self.accelerator.end_training()

    def log_videos(self):
        N = 1
        video_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            out=(self.temp_pred[:N] + 1) / 2,
            prompt=(log_txt_as_img((256, 256), self.batch_inputs.prompt[:N]) + 1) / 2,
        )
        if self.config.use_ema:
            # recompute for EMA results
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.collect_all_latents()
                ema_x = self.temp_stabilizer(ema_x)
                ema_x = self.decode_all_latents(ema_x)
                video_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in video_logs.items():
                    # Convert images (1, T, C, H, W) to (T, C, H, W)
                    images = images.squeeze(0)
                    images = images.float()
                    # Add video to tensorboard
                    tracker.writer.add_video(
                        f"video/{tag}",
                        images,
                        self.global_step,
                        fps=24,  # Adjust FPS
                    )

        for key, images in video_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .squeeze(0).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.output_dir, self.config.logging_dir, "log_videos", f"{self.global_step:07}", key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_images(self):
        N = 4
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
            prompt=(log_txt_as_img((256, 256), self.batch_inputs.prompt[:N]) + 1) / 2,
        )
        if self.config.use_ema:
            # recompute for EMA results
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_grads(self):
        self.unwrap_model(self.D).eval().requires_grad_(False)
        x = self.forward_generator()
        loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
        loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
        loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
        losses = [("l2", loss_l2), ("lpips", loss_lpips), ("disc", loss_disc)]
        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in self.config.log_grad_modules:
                    if module_name.endswith(suffix):
                        flat_grad = torch.cat([
                            p.grad.flatten() for p in module.parameters() if p.requires_grad
                        ])
                        lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v)).item()
            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, "checkpoints", f"checkpoint-{self.global_step}")
            # self.G.to("cpu")
            # self.G.save_lora_adapter(save_path)
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
            # self.G.to(self.device)
            # # Save ema weights
            # self.ema_handler.save_ema_weights(save_path)
            # logger.info(f"Saved ema weights to {save_path}")