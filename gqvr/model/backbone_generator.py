from typing import Literal, List, overload

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

from gqvr.utils.common import wavelet_reconstruction, make_tiled_fn
from gqvr.utils.tilevae import enable_tiled_vae


class BaseEnhancer:

    def __init__(
        self,
        base_model_path,
        weight_path,
        lora_modules,
        lora_rank,
        model_t,
        coeff_t,
        vae_cfg,
        device,
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.coeff_t = coeff_t
        self.vae_cfg = vae_cfg

        self.weight_dtype = torch.bfloat16
        self.device = device

    def init_models(self, init_vae=True):
        self.init_scheduler()
        self.init_text_models()
        if init_vae:
            self.init_vae()
        self.init_generator()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    @overload
    def init_vae(self):
        ...

    @overload
    def init_generator(self):
        ...

    @overload
    def prepare_inputs(self, batch_size, prompt):
        ...

    @overload
    def forward_generator(self, z_lq: torch.Tensor) -> torch.Tensor:
        ...
    
    @torch.no_grad()
    def enhance(self,
            lq: torch.Tensor,
            prompt: str,
            upscale: int = 1,
            return_type: Literal["pt", "np", "pil"] = "pt",
            only_vae_output=False,
            save_Gprocessed_latents=False,
            fname=None):
        
        bs = len(lq)
        
        # VAE encoding
        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        self.prepare_inputs(batch_size=bs, prompt=prompt)

        z_lq = self.vae.encode(lq.to(self.weight_dtype)).mode() 

        if only_vae_output:
            z = z_lq
        else:
            z = self.forward_generator(z_lq) # (N x 4 x 64 x 64) for (N, 3, 512, 512) input


        if save_Gprocessed_latents:
            assert fname != "", "No file name provided but save G-processed latents is on"
            torch.save(z.to(self.weight_dtype).detach().cpu(), fname)
            return 400

        x = self.vae.decode(z.to(self.weight_dtype)).float()

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]
    
    # BUG: Doesn't work with tiled vae right now
    @torch.no_grad()
    def enhance_tiled(
        self,
        lq: torch.Tensor,
        prompt: str,
        upscale: int = 1,
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        patch_size = 512

        # Prepare low-quality inputs
        bs = len(lq)
        lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        ref = lq
        h0, w0 = lq.shape[2:]
        if min(h0, w0) <= patch_size:
            lq = self.resize_at_least(lq, size=patch_size)

        # VAE encoding
        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq.shape[2:]
        # Pad vae input size to multiples of vae_scale_factor,
        # otherwise image size will be changed
        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)
        with enable_tiled_vae(
            self.vae,
            is_decoder=False,
            tile_size=patch_size,
            dtype=self.weight_dtype,
        ):
            z_lq = self.vae.encode(lq.to(self.weight_dtype)).mode()

        # Generator forward
        self.prepare_inputs(batch_size=bs, prompt=prompt)
        z = make_tiled_fn(
            fn=lambda z_lq_tile: self.forward_generator(z_lq_tile),
            size=patch_size // vae_scale_factor,
            stride=patch_size // 2 // vae_scale_factor,
            progress=True
        )(z_lq)
        with enable_tiled_vae(
            self.vae,
            is_decoder=True,
            tile_size=patch_size // vae_scale_factor,
            dtype=self.weight_dtype,
        ):
            x = self.vae.decode(z.to(self.weight_dtype)).float()
        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        x = wavelet_reconstruction(x, ref.to(device=self.device))

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]

    @staticmethod
    def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

    @staticmethod
    def resize_at_least(imgs: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = imgs.size()
        if h == w:
            new_h, new_w = size, size
        elif h < w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size
        return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)