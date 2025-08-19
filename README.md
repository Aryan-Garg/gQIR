# Generative QVR
<!-- Teaser Image/Video -->
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

## <a id="installation"></a>Installation

## <a id="quick_start"></a>Quick Start

## <a id="pretrained_models_and_dataset"></a>Pretrained Models and Dataset

## <a id="inference"></a>Inference 

> python3 infer_sd2GAN_stage2.py --config configs/inference/eval_sd2GAN.yaml

Keep changing the main function for now (make it arg passing only during deployment)

## <a id="training"></a>Training
<!-- Arch Image -->

#### Stage 1 - SPAD-CMOS Aligned VAE:
> accelerate launch train_daEncoder.py --config configs/train/train_daEncoder.yaml

#### Stage 2 - Adversarial Training with Diffusion Initialization:

> python3 train_sd2GAN.py --config configs/train/train_sd2gan.yaml

#### Stage 3 - Video Stabilization

**Precomputing latents:**

> conda activate hypir && cd apgi/gQVR
> python3 infer_sd2GAN_stage2.py --config configs/inference/eval_sd2GAN.yaml --device "cuda:7" --ds_txt dataset_txt_files/video_dataset_txt_files/combined_part07.txt

**Training Lifting Stage:**
> 

Currently the brightness scale/factor (proportional to PPP) is set to 1.0 for all simulations

## Stage 1: Training the Frame Denoiser/Degradation-Aware Encoder

### Stage 2: Finetuning the Generative Pipeline with it

### Stage 3: Conversion to a Video Handling Prior


## <a id="citation"></a>Citation
Please cite us if our work is useful for your research.

<!-- TODO: Replace with conference bibtex after publication -->

```bibtex
@misc{garg2025_gqvr
    title={gQVR: Generative Quanta Video Restoration},
    author={Aryan Garg and Sizhuo Ma and Mohit Gupta},
    year={2025},
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