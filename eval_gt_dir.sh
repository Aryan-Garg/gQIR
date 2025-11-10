#!/bin/bash

# S2:

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_explosion_0001_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_jetengine_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_moreguns_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_padlock_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_tank_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00014_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00020_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00021_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00031_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/xvfi_train_gt 

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/xvfi_boat_gt 

# S1:

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_explosion_0001_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_jetengine_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_moreguns_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_padlock_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/small_tank_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00014_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00020_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00021_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/test_00031_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/xvfi_train_gt --only_vae

python3 infer_sd2GAN_stage2.py --config configs/inference/eval_3bit_mono.yaml \
 --eval_gt_dir --gt_dir /nobackup1/aryan/results/sd21_burst/xvfi_boat_gt --only_vae