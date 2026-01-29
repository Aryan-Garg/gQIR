import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

from .backbone_generator import BaseEnhancer
from gqvr.model.vae import AutoencoderKL
from fvcore.nn import FlopCountAnalysis

class SD2Enhancer(BaseEnhancer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)
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

    def init_vae(self):
        self.vae = AutoencoderKL(self.vae_cfg.ddconfig, self.vae_cfg.embed_dim)       
        # Load quanta VAE
        daVAE = torch.load(self.vae_cfg.qvae_path, map_location="cpu")
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
        self.vae.to(device=self.device).to(self.weight_dtype)
        vae_unused = set(daVAE.keys()) - vae_used
        
        if len(vae_unused) == 0:
            print(f"[+] Loaded qVAE successfully from {self.vae_cfg.qvae_path}\n")
        else:
            print(f"[!] VAE keys NOT used: {vae_unused}\n")

        self.vae.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        txt_ids = self.tokenizer(
            [prompt] * bs,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]
        c_txt = {"text_embed": text_embed}
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

    def forward_generator(self, z_lq):
        z_in = z_lq * 0.18215 # Use self.vae.config.scaling_factor if using diffusers
        # Output: UNet flops 531320586240
        # flops_unet = FlopCountAnalysis(self.G, (z_in, self.inputs["timesteps"], self.inputs["c_txt"]["text_embed"]))
        # print('UNet flops', flops_unet.total())

        eps = self.G(
            z_in, 
            self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        z_out = z.to(self.weight_dtype) / 0.18215 # VAE's scaling factor
        return z_out