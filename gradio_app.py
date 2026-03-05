#!/usr/bin/env python3
"""gQIR Gradio demo

Single-frame mode follows infer_sd2GAN_stage2.py (color path only).
Burst mode follows infer_burst_realistic.py for 77->11 aggregation and reconstruction.

Local run cmd:
python gradio_app.py 
  --single-config configs/inference/eval_sd2GAN.yaml \
  --burst-config configs/inference/eval_burst_mosaic.yaml \
  --device cuda --local
"""

from __future__ import annotations

import argparse
import atexit
import os
import random
import shutil
import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf
from peft import LoraConfig
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from gqvr.dataset.utils import emulate_spc, srgb_to_linearrgb
from gqvr.model.core_raft.raft import RAFT
from gqvr.model.fusionViT import LightweightHybrid3DFusion
from gqvr.model.generator import SD2Enhancer
from gqvr.model.vae import AutoencoderKL

try:
    import h5py
except Exception:
    h5py = None


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_SINGLE_CONFIG_COLOR = APP_ROOT / "configs" / "inference" / "eval_3bit_color.yaml"
DEFAULT_SINGLE_CONFIG_MONO = APP_ROOT / "configs" / "inference" / "eval_3bit_mono.yaml"
DEFAULT_BURST_CONFIG_COLOR = APP_ROOT / "configs" / "inference" / "eval_burst_mosaic.yaml"
DEFAULT_BURST_CONFIG_MONO = APP_ROOT / "configs" / "inference" / "eval_burst.yaml"
DEFAULT_MAX_SIZE = (512, 512)
BURST_WINDOW = 77

PIPELINE_COLOR = "Color"
PIPELINE_MONO = "Monochrome"
PIPELINE_OPTIONS = [PIPELINE_COLOR, PIPELINE_MONO]

SINGLE_MODE_GT = "GT image (simulate 3-bit SPAD)"
SINGLE_MODE_REAL = "Real SPAD frame"
BURST_MODE_GT = "GT cube (simulate SPAD from RGB cube)"
BURST_MODE_REAL = "Real photon cube / SPAD cube"

TO_TENSOR = transforms.ToTensor()

_SINGLE_PIPELINES: dict[str, "SingleColorPipeline"] = {}
_BURST_PIPELINES: dict[str, "BurstColorPipeline"] = {}
_SINGLE_LOCK = threading.Lock()
_BURST_LOCK = threading.Lock()

RUNTIME_SINGLE_CONFIGS: dict[str, Path] = {}
RUNTIME_BURST_CONFIGS: dict[str, Path] = {}
RUNTIME_DEVICE: str = "cpu"
RUNTIME_BURST_OUT_SIZES: dict[str, int] = {}
_TEMP_VIDEO_DIRS: list[str] = []


def _cleanup_temp_video_dirs() -> None:
    for p in _TEMP_VIDEO_DIRS:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_temp_video_dirs)


@dataclass
class CubeDescriptor:
    source_mode: str
    kind: str  # dir | video | array | h5
    path: str
    total_frames: int
    out_size: int
    files: Optional[list[str]] = None
    array_format: Optional[str] = None  # npy | npz | pt
    array_key: Optional[str] = None
    h5_keys: Optional[list[str]] = None
    temp_dir: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-config",
        type=str,
        default=None,
        help="Deprecated alias for --single-config-color",
    )
    parser.add_argument(
        "--burst-config",
        type=str,
        default=None,
        help="Deprecated alias for --burst-config-color",
    )
    parser.add_argument("--single-config-color", type=str, default=str(DEFAULT_SINGLE_CONFIG_COLOR))
    parser.add_argument("--single-config-mono", type=str, default=str(DEFAULT_SINGLE_CONFIG_MONO))
    parser.add_argument("--burst-config-color", type=str, default=str(DEFAULT_BURST_CONFIG_COLOR))
    parser.add_argument("--burst-config-mono", type=str, default=str(DEFAULT_BURST_CONFIG_MONO))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda, cuda:0, cpu",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--local", action="store_true", help="Bind to 127.0.0.1 instead of 0.0.0.0")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _ensure_rgb_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxWx3, got {arr.shape}")
    return arr


def _normalize_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(np.float32)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if 0.0 <= min_v and max_v <= 1.0:
        return arr
    if min_v >= 0.0 and max_v <= 255.0:
        arr = arr / 255.0
    elif min_v >= 0.0 and max_v > 0.0:
        arr = arr / max_v
    else:
        den = max(max_v - min_v, 1e-8)
        arr = (arr - min_v) / den
    return np.clip(arr, 0.0, 1.0)


def _to_uint8(arr_float01: np.ndarray) -> np.ndarray:
    return np.clip(arr_float01 * 255.0, 0.0, 255.0).astype(np.uint8)


def _resize_dims_keep_aspect(h: int, w: int, max_side: int, multiple_of: int = 1) -> tuple[int, int]:
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame size: {h}x{w}")
    scale = float(max_side) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if multiple_of > 1:
        new_h = max(multiple_of, int(round(new_h / multiple_of) * multiple_of))
        new_w = max(multiple_of, int(round(new_w / multiple_of) * multiple_of))
    new_h = min(new_h, max_side)
    new_w = min(new_w, max_side)
    return new_h, new_w


def _resize_frame_rgb(frame_float01: np.ndarray, max_side: int, multiple_of: int = 1) -> np.ndarray:
    frame_float01 = _normalize_float01(_ensure_rgb_image(frame_float01))
    h, w = frame_float01.shape[:2]
    new_h, new_w = _resize_dims_keep_aspect(h, w, max_side=max_side, multiple_of=multiple_of)
    if h == new_h and w == new_w:
        return frame_float01
    pil_img = Image.fromarray(_to_uint8(frame_float01))
    return np.asarray(pil_img.resize((new_w, new_h), Image.LANCZOS), dtype=np.float32) / 255.0


def _resize_frames_rgb(frames_thwc: np.ndarray, max_side: int, multiple_of: int = 1) -> np.ndarray:
    frames_thwc = np.asarray(frames_thwc)
    if frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
        raise ValueError(f"Expected THWC with C=3, got {frames_thwc.shape}")

    h, w = frames_thwc.shape[1:3]
    new_h, new_w = _resize_dims_keep_aspect(h, w, max_side=max_side, multiple_of=multiple_of)
    if h == new_h and w == new_w:
        return _normalize_float01(frames_thwc)

    resized = [
        _resize_frame_rgb(frames_thwc[i], max_side=max_side, multiple_of=multiple_of)
        for i in range(frames_thwc.shape[0])
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def _to_gray_uint8(img_uint8_rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img_uint8_rgb is None:
        return None
    arr = np.asarray(img_uint8_rgb)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.uint8)
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("L"), dtype=np.uint8)


def _resize_uint8_to_hw(img_uint8: Optional[np.ndarray], target_h: int, target_w: int) -> Optional[np.ndarray]:
    if img_uint8 is None:
        return None
    arr = np.asarray(img_uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.uint8), mode="L")
        return np.asarray(pil.resize((target_w, target_h), Image.LANCZOS), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        pil = Image.fromarray(arr[..., 0].astype(np.uint8), mode="L")
        return np.asarray(pil.resize((target_w, target_h), Image.LANCZOS), dtype=np.uint8)
    pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize((target_w, target_h), Image.LANCZOS), dtype=np.uint8)


def _to_thwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            out = arr
        elif arr.shape[1] in (1, 3, 4):
            out = np.transpose(arr, (0, 2, 3, 1))
        elif arr.shape[0] in (1, 3, 4):
            out = np.transpose(arr, (3, 1, 2, 0))
        else:
            raise ValueError(f"Cannot infer channel axis from shape {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            out = arr[None, ...]
        elif arr.shape[0] in (1, 3, 4):
            out = np.transpose(arr, (1, 2, 0))[None, ...]
        else:
            # Treat as T x H x W single-channel.
            out = arr[..., None]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

    if out.shape[-1] == 4:
        out = out[..., :3]
    return out


def _single_channel_bayer_to_sparse_rgb(frames_thw1: np.ndarray) -> np.ndarray:
    bayer = np.asarray(frames_thw1).astype(np.float32)
    if bayer.ndim != 4 or bayer.shape[-1] != 1:
        raise ValueError(f"Expected THW1, got {bayer.shape}")
    bayer = bayer[..., 0]
    t, h, w = bayer.shape
    out = np.zeros((t, h, w, 3), dtype=np.float32)
    out[:, 0::2, 0::2, 0] = bayer[:, 0::2, 0::2]
    out[:, 0::2, 1::2, 1] = bayer[:, 0::2, 1::2]
    out[:, 1::2, 0::2, 1] = bayer[:, 1::2, 0::2]
    out[:, 1::2, 1::2, 2] = bayer[:, 1::2, 1::2]
    return out


def _mosaic_with_pattern(img_rgb: np.ndarray, pattern: str) -> np.ndarray:
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    out = np.zeros_like(img_rgb, dtype=np.float32)

    if pattern == "RGGB":
        out[0::2, 0::2, 0] = r[0::2, 0::2]
        out[0::2, 1::2, 1] = g[0::2, 1::2]
        out[1::2, 0::2, 1] = g[1::2, 0::2]
        out[1::2, 1::2, 2] = b[1::2, 1::2]
    elif pattern == "GRBG":
        out[0::2, 1::2, 0] = r[0::2, 1::2]
        out[0::2, 0::2, 1] = g[0::2, 0::2]
        out[1::2, 1::2, 1] = g[1::2, 1::2]
        out[1::2, 0::2, 2] = b[1::2, 0::2]
    elif pattern == "BGGR":
        out[0::2, 0::2, 2] = b[0::2, 0::2]
        out[0::2, 1::2, 1] = g[0::2, 1::2]
        out[1::2, 0::2, 1] = g[1::2, 0::2]
        out[1::2, 1::2, 0] = r[1::2, 1::2]
    elif pattern == "GBRG":
        out[0::2, 0::2, 1] = g[0::2, 0::2]
        out[1::2, 1::2, 1] = g[1::2, 1::2]
        out[0::2, 1::2, 2] = b[0::2, 1::2]
        out[1::2, 0::2, 0] = r[1::2, 0::2]
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    return out


def _simulate_single_3bit_from_gt(gt_rgb_float01: np.ndarray, target_ppp: float) -> np.ndarray:
    bits = 3
    n = (2**bits) - 1
    factor = target_ppp / 3.5
    lq_sum = np.zeros_like(gt_rgb_float01, dtype=np.float32)
    for _ in range(n):
        spc = emulate_spc(srgb_to_linearrgb(gt_rgb_float01), factor=factor).astype(np.float32)
        pattern = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])
        lq_sum += _mosaic_with_pattern(spc, pattern)
    return np.clip(lq_sum / float(n), 0.0, 1.0)


def _simulate_single_3bit_from_gt_mono(gt_rgb_float01: np.ndarray, target_ppp: float) -> np.ndarray:
    bits = 3
    n = (2**bits) - 1
    factor = target_ppp / 3.5
    lq_sum = np.zeros_like(gt_rgb_float01, dtype=np.float32)
    for _ in range(n):
        spc = emulate_spc(srgb_to_linearrgb(gt_rgb_float01), factor=factor).astype(np.float32)
        lq_sum += spc
    return np.clip(lq_sum / float(n), 0.0, 1.0)


def _simulate_binary_burst_frame_from_gt(gt_rgb_float01: np.ndarray, target_ppp: float = 3.5) -> np.ndarray:
    # Same PPP scaling convention as single-frame simulation.
    factor = target_ppp / 3.5
    spc = emulate_spc(srgb_to_linearrgb(gt_rgb_float01), factor=factor).astype(np.float32)
    return _mosaic_with_pattern(spc, "BGGR")


def _simulate_binary_burst_frame_from_gt_mono(gt_rgb_float01: np.ndarray, target_ppp: float = 3.5) -> np.ndarray:
    # Same PPP scaling convention as single-frame simulation.
    factor = target_ppp / 3.5
    return emulate_spc(srgb_to_linearrgb(gt_rgb_float01), factor=factor).astype(np.float32)


def _tensor_to_uint8_image(x_bchw: torch.Tensor) -> np.ndarray:
    x = x_bchw.detach().cpu().clamp(0.0, 1.0)
    x = (x[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return x


def _encode_prompt(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompt: str, bs: int, device: str) -> torch.Tensor:
    txt_ids = tokenizer(
        [prompt] * bs,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return text_encoder(txt_ids.to(device))[0]


def differentiable_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    flow = flow.permute(0, 2, 3, 1)
    new_grid = grid + flow
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (w - 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (h - 1) - 1.0
    return F.grid_sample(x, new_grid, align_corners=True, padding_mode="border")


class SingleColorPipeline:
    def __init__(self, config_path: Path, device: str):
        self.config_path = config_path
        self.device = device
        self.max_size = DEFAULT_MAX_SIZE
        self.model: Optional[SD2Enhancer] = None

    def load(self) -> None:
        if self.model is not None:
            return
        cfg = OmegaConf.load(str(self.config_path))
        if cfg.base_model_type != "sd2":
            raise ValueError(f"Unsupported base_model_type for single pipeline: {cfg.base_model_type}")
        self.model = SD2Enhancer(
            base_model_path=cfg.base_model_path,
            weight_path=cfg.weight_path,
            lora_modules=cfg.lora_modules,
            lora_rank=cfg.lora_rank,
            model_t=cfg.model_t,
            coeff_t=cfg.coeff_t,
            vae_cfg=cfg.model.vae_cfg,
            device=self.device,
        )
        self.model.init_models()

    def _enhance(self, lq_rgb_float01: np.ndarray, prompt: str, only_vae_output: bool, seed: int) -> tuple[np.ndarray, int]:
        if self.model is None:
            self.load()
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)

        out_h, out_w = lq_rgb_float01.shape[:2]
        if out_h * out_w > self.max_size[0] * self.max_size[1]:
            raise ValueError(
                f"Resolution {out_h}x{out_w} exceeds max pixel budget "
                f"{self.max_size[0]}x{self.max_size[1]}."
            )

        image_tensor = TO_TENSOR(lq_rgb_float01).unsqueeze(0)
        pil_img = self.model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=1,
            return_type="pil",
            only_vae_output=only_vae_output,
            save_Gprocessed_latents=False,
            fname="",
        )[0]
        return np.asarray(pil_img.convert("RGB"), dtype=np.uint8), seed

    def reconstruct_from_gt(
        self,
        gt_image_np: np.ndarray,
        prompt: str,
        target_ppp: float,
        only_vae_output: bool,
        seed: int,
        simulate_color_mosaic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        gt_rgb = _normalize_float01(_ensure_rgb_image(gt_image_np))
        gt_rgb = _resize_frame_rgb(gt_rgb, self.max_size[0], multiple_of=8)
        if simulate_color_mosaic:
            lq_rgb = _simulate_single_3bit_from_gt(gt_rgb, target_ppp=target_ppp)
        else:
            lq_rgb = _simulate_single_3bit_from_gt_mono(gt_rgb, target_ppp=target_ppp)
        recon_uint8, used_seed = self._enhance(lq_rgb, prompt, only_vae_output, seed)
        status = f"Single reconstruction complete (mode=GT simulation, seed={used_seed}, PPP={target_ppp:.2f})."
        return _to_uint8(gt_rgb), _to_uint8(lq_rgb), recon_uint8, status

    def reconstruct_from_real_spad(
        self,
        lq_image_np: np.ndarray,
        prompt: str,
        only_vae_output: bool,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        lq_rgb = _normalize_float01(_ensure_rgb_image(lq_image_np))
        lq_rgb = _resize_frame_rgb(lq_rgb, self.max_size[0], multiple_of=8)
        recon_uint8, used_seed = self._enhance(lq_rgb, prompt, only_vae_output, seed)
        status = f"Single reconstruction complete (mode=real SPAD frame, seed={used_seed})."
        return _to_uint8(lq_rgb), recon_uint8, status


class BurstColorPipeline:
    def __init__(self, config_path: Path, device: str):
        self.config_path = config_path
        self.device = device
        self.cfg = None
        self.out_size = 512
        self.weight_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        self.vae: Optional[AutoencoderKL] = None
        self.raft_model: Optional[RAFT] = None
        self.fusion_vit: Optional[LightweightHybrid3DFusion] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.ls_burst_unet: Optional[UNet2DConditionModel] = None

    def load(self) -> None:
        if self.vae is not None:
            return

        cfg = OmegaConf.load(str(self.config_path))
        self.cfg = cfg
        self.out_size = int(cfg.dataset.val.params.out_size)

        vae = AutoencoderKL(cfg.model.vae_cfg.ddconfig, cfg.model.vae_cfg.embed_dim)
        da_vae = torch.load(cfg.qvae_path, map_location="cpu")
        init_vae = {}
        scratch = vae.state_dict()
        for key in scratch:
            if key in da_vae:
                init_vae[key] = da_vae[key].clone()
        vae.load_state_dict(init_vae, strict=True)
        vae.requires_grad_(False)
        vae.eval().to(self.device)
        self.vae = vae

        class RAFTArgs:
            mixed_precision = True
            small = False
            alternate_corr = True
            dropout = False

        raft_model = RAFT(RAFTArgs())
        raft_path = APP_ROOT / "pretrained_ckpts" / "models" / "raft-things.pth"
        raft_dict = torch.load(str(raft_path), map_location="cpu")
        corrected = {}
        for k, v in raft_dict.items():
            k2 = ".".join(k.split(".")[1:]) if "." in k else k
            corrected[k2] = v
        raft_model.load_state_dict(corrected)
        raft_model.eval().requires_grad_(False).to(self.device)
        self.raft_model = raft_model

        fusion_vit = LightweightHybrid3DFusion()
        fusion_ckpt = torch.load(cfg.fusion_vit_weight_path, map_location="cpu")
        fusion_vit.load_state_dict(fusion_ckpt)
        fusion_vit.eval().requires_grad_(False).to(self.device)
        self.fusion_vit = fusion_vit

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(cfg.base_model_path, subfolder="scheduler")
        ls_burst_unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=cfg.lora_modules,
        )
        ls_burst_unet.add_adapter(lora_cfg)

        try:
            state_dict = torch.load(cfg.unet_weight_path, map_location="cpu", weights_only=False)
        except TypeError:
            state_dict = torch.load(cfg.unet_weight_path, map_location="cpu")

        ls_burst_unet.load_state_dict(state_dict, strict=False)
        required_keys = {k for k in ls_burst_unet.state_dict().keys() if "lora" in k}
        input_keys = set(state_dict.keys())
        if required_keys != input_keys:
            missing = required_keys - input_keys
            unexpected = input_keys - required_keys
            raise RuntimeError(f"LoRA key mismatch. Missing={len(missing)} Unexpected={len(unexpected)}")

        ls_burst_unet.eval().requires_grad_(False)
        self.ls_burst_unet = ls_burst_unet

    def _ensure_loaded(self) -> None:
        if self.vae is None:
            self.load()

    def reconstruct_from_binary_window(
        self,
        binary_window_77: np.ndarray,
        gt_window_77: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self._ensure_loaded()
        assert self.cfg is not None
        assert self.vae is not None
        assert self.raft_model is not None
        assert self.fusion_vit is not None
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        assert self.scheduler is not None
        assert self.ls_burst_unet is not None

        if binary_window_77.shape[0] != BURST_WINDOW:
            raise ValueError(f"Burst window must have {BURST_WINDOW} frames, got {binary_window_77.shape[0]}")

        binary_window_77 = _resize_frames_rgb(
            _normalize_float01(binary_window_77),
            max_side=self.out_size,
            multiple_of=64,
        )

        lqs = torch.from_numpy(binary_window_77).unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
        lqs = (lqs * 2.0) - 1.0

        lqs_3bit = []
        for i in range(0, lqs.size(1), 7):
            chunk = lqs[:, i : i + 7, ...]
            if chunk.size(1) < 7:
                break
            lqs_3bit.append(torch.mean(chunk, dim=1, keepdim=True))
        lqs = torch.cat(lqs_3bit, dim=1)

        gts = None
        if gt_window_77 is not None:
            gt_window_77 = _resize_frames_rgb(
                _normalize_float01(gt_window_77),
                max_side=self.out_size,
                multiple_of=64,
            )
            gts = torch.from_numpy(gt_window_77).unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.device)
            gts = (gts * 2.0) - 1.0
            gts_3bit = []
            for i in range(0, gts.size(1), 7):
                chunk = gts[:, i : i + 7, ...]
                if chunk.size(1) < 7:
                    break
                gts_3bit.append(torch.mean(chunk, dim=1, keepdim=True))
            gts = torch.cat(gts_3bit, dim=1)

        with torch.inference_mode():
            bs = lqs.size(0)
            t_total = lqs.size(1)
            center_t = t_total // 2

            latents = []
            decoded_lqs = []
            for t in range(t_total):
                lq_t = lqs[:, t, ...].float()
                z_t = self.vae.encode(lq_t).mode()
                latents.append(z_t)
                decoded_lqs.append(self.vae.decode(z_t).float())

            y = torch.stack(decoded_lqs, dim=1)
            flow_vectors = []
            for t in range(t_total):
                ls_in = y[:, t, ...].float()
                center_in = y[:, center_t, ...].float()
                if t < center_t:
                    _, flow_bw = self.raft_model(center_in, ls_in, iters=20, test_mode=True)
                else:
                    _, flow_bw = self.raft_model(ls_in, center_in, iters=20, test_mode=True)
                z_h, z_w = latents[t].shape[-2:]
                in_h, in_w = ls_in.shape[-2:]
                flow_bw = F.interpolate(flow_bw, size=(z_h, z_w), mode="bilinear", align_corners=True)
                flow_bw[:, 0] *= float(z_w) / float(in_w)
                flow_bw[:, 1] *= float(z_h) / float(in_h)
                flow_vectors.append(flow_bw)

            aligned_latents = []
            for t in range(t_total):
                latent_t = latents[t]
                if t == center_t:
                    aligned_latents.append(latent_t)
                else:
                    aligned_latents.append(differentiable_warp(latent_t, flow_vectors[t]))

            aligned_latents = torch.stack(aligned_latents, dim=1)
            merged_latent = self.fusion_vit(aligned_latents)

            z_in = (merged_latent * 0.18215).to(self.weight_dtype)
            timesteps = torch.full((bs,), int(self.cfg.model_t), dtype=torch.long, device=self.device)
            text_embed = _encode_prompt(self.tokenizer, self.text_encoder, "", bs=bs, device=self.device)
            eps = self.ls_burst_unet(z_in, timesteps, encoder_hidden_states=text_embed).sample
            z = self.scheduler.step(eps, int(self.cfg.coeff_t), z_in).pred_original_sample
            decoded_refined = self.vae.decode(z.float() / 0.18215).float().clamp(0.0, 1.0)

            center_input = ((lqs[:, center_t, ...] + 1.0) / 2.0).clamp(0.0, 1.0)
            center_gt = None
            if gts is not None:
                center_gt = ((gts[:, center_t, ...] + 1.0) / 2.0).clamp(0.0, 1.0)

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        return (
            _tensor_to_uint8_image(center_input),
            _tensor_to_uint8_image(decoded_refined),
            _tensor_to_uint8_image(center_gt) if center_gt is not None else None,
        )


def _get_single_pipeline(pipeline_type: str) -> SingleColorPipeline:
    if pipeline_type not in PIPELINE_OPTIONS:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    if pipeline_type not in RUNTIME_SINGLE_CONFIGS:
        raise RuntimeError(f"Single config not initialized for pipeline type: {pipeline_type}")
    with _SINGLE_LOCK:
        if pipeline_type not in _SINGLE_PIPELINES:
            _SINGLE_PIPELINES[pipeline_type] = SingleColorPipeline(
                RUNTIME_SINGLE_CONFIGS[pipeline_type],
                RUNTIME_DEVICE,
            )
            _SINGLE_PIPELINES[pipeline_type].load()
    return _SINGLE_PIPELINES[pipeline_type]


def _get_burst_pipeline(pipeline_type: str) -> BurstColorPipeline:
    if pipeline_type not in PIPELINE_OPTIONS:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    if pipeline_type not in RUNTIME_BURST_CONFIGS:
        raise RuntimeError(f"Burst config not initialized for pipeline type: {pipeline_type}")
    with _BURST_LOCK:
        if pipeline_type not in _BURST_PIPELINES:
            _BURST_PIPELINES[pipeline_type] = BurstColorPipeline(
                RUNTIME_BURST_CONFIGS[pipeline_type],
                RUNTIME_DEVICE,
            )
            _BURST_PIPELINES[pipeline_type].load()
    return _BURST_PIPELINES[pipeline_type]


def _get_runtime_burst_out_size(pipeline_type: str) -> int:
    if pipeline_type not in RUNTIME_BURST_OUT_SIZES:
        return DEFAULT_MAX_SIZE[0]
    return int(RUNTIME_BURST_OUT_SIZES[pipeline_type])


def _resolve_uploaded_path(uploaded_file: Any, local_path: str) -> Optional[str]:
    if isinstance(uploaded_file, str) and uploaded_file:
        return uploaded_file
    if hasattr(uploaded_file, "name") and uploaded_file.name:
        return uploaded_file.name
    if isinstance(uploaded_file, dict) and uploaded_file.get("name"):
        return uploaded_file["name"]

    if local_path and local_path.strip():
        return local_path.strip()
    return None


def _list_image_files(dir_path: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for name in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and Path(name).suffix.lower() in exts:
            files.append(p)
    return files


def _is_video_extension(ext: str) -> bool:
    # Include common public-facing formats. Keep ".wav" for user compatibility;
    # extraction will fail with a clear message since it is audio-only.
    return ext in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".wav"}


def _extract_video_frames_to_temp(video_path: str) -> tuple[str, list[str]]:
    temp_dir = tempfile.mkdtemp(prefix="gqir_video_frames_")
    out_pattern = os.path.join(temp_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
    