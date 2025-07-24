"""
All models used in inference:
- gQVR-v1
  All tasks share the same pre-trained stable diffusion 2.1 ZSNR LAION-5 Aesthetic.
-- Image Restoration/Denoising task
    stage-1 model (N2N): N2N trained on our curated dataset.
    stage-2 model (v1): SPADEd + LIEM + Sampling-Guided ControlNet 
-- Video Restoration task
    stage-1 model (N2N): N2N trained on our curated dataset.
    stage-2 model (v1): SPADEd + LIEM + Sampling-Guided ControlNet trained 
    stage-3 model (v1-motion): RAFT optical flow for temporal consistency + TSA-blocks + input bursts for v1  
"""
MODELS = {
    # --------------- stage-1 model weights ---------------
    "n2n": "",
    # --------------- pre-trained stable diffusion weights ---------------
    "sd_v2.1_zsnr": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
    # --------------- controlnet weights ---------------
    "v1": "",
    # --------------- burst controlnet weights ---------------
    "v1-motion": "",
}
