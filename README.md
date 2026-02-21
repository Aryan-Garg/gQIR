# [CVPR 2026] gQIR: Generative Quanta Image Reconstruction

<!-- TODO: Teaser Image/Video -->
![color_spads](assets/README_teaser_color_SPAD.png)

![Static Badge](https://img.shields.io/badge/🐧-project_page-green?link=https%3A%2F%2Faryan-garg.github.io%2Fgqvr)
[![arXiv](https://img.shields.io/badge/arXiv-TO.REPLACE-b31b1b.svg)](https://arxiv.org/abs/TO.REPLACE) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/AryanGarg/TODO) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Aryan-Garg/gQVR)

[Aryan Garg](https://aryan-garg.github.io/)<sup>1</sup>, [Sizhuo Ma](https://sizhuoma.netlify.app/)<sup>2</sup>, [Mohit Gupta](https://wisionlab.com/people/mohit-gupta/)<sup>1</sup>,

<sup>1</sup> University of Wisconsin-Madison<br><sup>2</sup> SnapChat, USA<br>


## Table of Contents

- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick_start)
- [Pretrained Models and Dataset](#pretrained_models_and_dataset)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## <a id="results"></a>Results 

TODO

## <a id="installation"></a>Installation

TODO

## <a id="quick_start"></a>Quick Start

TODO

## <a id="pretrained_models_and_dataset"></a>Pretrained Models and Dataset

TODO

## <a id="inference"></a>Inference 

> python3 infer_sd2GAN_stage2.py --config configs/inference/eval_sd2GAN.yaml 


#### For real world captures:

> python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_color.yaml --ds_txt ds_txt_real_captures.txt --real_captures


## <a id="training"></a>Training
<!-- Arch Image -->

#### Stage 1 - SPAD-CMOS Aligned VAE:

> conda activate hypir
> accelerate launch train_daEncoder.py --config configs/train/train_daEncoder.yaml
> CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 accelerate launch --main_process_port 29502 train_s1_mosaic.py --config configs/train/train_s1_mosaic_1bit.yaml
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29503 train_s1_mosaic.py --config configs/train/train_s1_mosaic_3bit.yaml

#### Stage 2 - Latent Space Enhancement - Adversarial Training with Diffusion Initialization:

> python3 train_sd2GAN.py --config configs/train/train_sd2gan.yaml

#### Stage 3 - Burst Processing - Fidelity Upgrade

![latent_visualization](assets/latent_viz.png)

Key insight: Second channel of the encoded latents have the most semantic information which enables flow warping to the center channel for burst refinement. However 4x upscaling is still required to compute flow.

> CUDA_VISIBLE_DEVICES=1 python3 train_burst.py --config configs/train/train_burst.yaml (for comparison with QUIVER & QBP)
> CUDA_VISIBLE_DEVICES=1 python3 train_burst.py --config configs/train/train_burst_mosaic.yaml (color-burst model)


**Precomputing latents:**

> conda activate hypir && cd apgi/gQVR
> python3 infer_sd2GAN_stage2.py --config configs/inference/eval_sd2GAN.yaml --device "cuda:0" --ds_txt dataset_txt_files/video_dataset_txt_files/combined_part00.txt

**1. Training Latent Space 3D Stabilizer:**

> conda activate hypir && cd apgi/gQVR
> accelerate launch train_3Dtemp_stabilization.py --config configs/train/train_sd2gan_video.yaml

**2. Training ConvEMA Decoder:**
> conda activate hypir && cd apgi/gQVR
> accelerate launch train_convEMADecoder.py --config configs/train/train_convEMA_decoder_stage3.yaml

---

Currently the brightness scale/factor (proportional to PPP) is set to 1.0 for all simulations


## <a id="citation"></a>Citation
Please cite us if our work is useful for your research.

<!-- TODO: Replace with conference bibtex after publication -->

```bibtex
@misc{garg2026_gqvr
    title={gQVR: Generative Quanta Image Reconstruction},
    author={Garg, Aryan and Ma, Sizhuo and  Gupta, Mohit},
    year={2026},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License 


## <a id="acknowledgements"></a>Acknowledgements

This project is based on [XPixelGroup](https://xpixel.group/)'s projects: [DiffBIR](https://github.com/XPixelGroup/DiffBIR) and [HYPIR](https://github.com/XPixelGroup/HYPIR). Thanks for their amazing work.

## <a name="contact"></a>Contact

If you have any questions, please feel free to contact with me at [agarg54@wisc.edu](mailto:agarg54@wisc.edu).