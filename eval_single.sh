#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 infer_sd2GAN_stage2.py --config "configs/inference/eval_3bit_color.yaml" \
 --eval_single_image --single_img_path "/media/agarg54/Extreme SSD/YouHQ/YouHQ-Extracted/animal/Y5_6LALJ_mc/063600_063659_00/frame_0000.png"