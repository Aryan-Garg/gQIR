from typing import List, Dict
import os
import torch
from accelerate.logging import get_logger
from peft import LoraConfig
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from argparse import ArgumentParser
from omegaconf import OmegaConf

from gqvr.model.minivit import miniViT_3D
from base_trainer import BaseTrainer


logger = get_logger(__name__, log_level="INFO")


class ipLiftingTrainer(BaseTrainer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.config.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def init_generator(self):
        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
           self.base_model_path, subfolder="unet", torch_dtype=self.weight_dtype).to(self.device)
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_rank,
           init_lora_weights="gaussian", target_modules=target_modules)
        self.G.add_adapter(G_lora_cfg)

        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        self.G.load_state_dict(state_dict, strict=False)
        input_keys = set(state_dict.keys())
        required_keys = set([k for k in self.G.state_dict().keys() if "lora" in k])
        missing = required_keys - input_keys
        unexpected = input_keys - required_keys
        assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"

        self.G.eval().requires_grad_(False)

    def init_minivit_stabilizer(self):
        self.minivit_stabilizer = miniViT_3D(
            in_channels=self.config.model.minivit_cfg.in_max_frames,
            z_channels=self.config.model.minivit_cfg.z_channels,
            z_resolution=self.config.model.minivit_cfg.z_resolution
        ).to(self.device)
        self.minivit_stabilizer.train().requires_grad_(True)

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        txt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.accelerator.device))[0]
        return {"text_embed": text_embed}
        
    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, miniViT_3D)
                state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.detach().clone().data
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            assert isinstance(model, miniViT_3D), "Model must be an instance of miniViT_3D"
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"))
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loading miniViT_3D parameters, unexpected keys: {u}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def collect_all_latents(self):
        zs = []
        for i in range(len(self.batch_inputs.z_lq.size(1))):
            this_z_lq = self.batch_inputs.z_lq[:, i, ...]
            z_in = this_z_lq * 0.18215 # vae scaling factor
            eps = self.G(
                z_in,
                self.batch_inputs.timesteps,
                encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
            ).sample
            z = self.scheduler.step(eps, self.config.coeff_t, z_in).pred_original_sample
            zs.append(z)
        z = torch.stack(zs, dim=1)
        return z
    
    def forward_minivit(self, z: torch.Tensor) -> torch.Tensor:
        stable_zs = self.minivit_stabilizer(z)
        xs = []
        for i in range(len(stable_zs.size(1))):
            z = stable_zs[:, i, ...]
            # Decode the latent z using the VAE
            x = self.vae.decode(z.to(self.weight_dtype) / 0.18215).float()
            xs.append(x)
        x = torch.stack(xs, dim=1)
        return x
        
    

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)

if config.base_model_type == "sd2":
    trainer = ipLiftingTrainer(config)
    trainer.run()
else:
    raise ValueError(f"Unsupported model type: {config.base_model_type}")