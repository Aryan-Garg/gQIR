#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 infer_sd2GAN_stage2.py --config "configs/inference/eval_3bit_color.yaml" \
 --real_captures --ds_txt ds_txt_real_capture_die.txt