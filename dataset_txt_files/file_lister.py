#!/usr/bin/env python3
import math
import os
import sys
from tqdm import tqdm
import numpy as np
import argparse


def generate_file_list_for_i2(base_path):
    for sub_dir in tqdm(os.listdir(base_path)):
        sub_dir_path = os.path.join(base_path, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for img in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img)
            if not img.endswith('.png'):
                continue
            if "test" in sub_dir:
                with open("i2k2_test.txt", 'a') as f:
                    f.write(f"{img_path}\n")
            else:
                with open("i2k2_train.txt", 'a') as f:
                    f.write(f"{img_path}\n")


def generate_file_list_for_visionsim(base_path):
    for scene_name in tqdm(os.listdir(base_path)):
        scene_path = os.path.join(base_path, scene_name)
        if not os.path.isdir(scene_path):
            continue
        for sub_dir in os.listdir(scene_path):
            if "frames" not in sub_dir and "spc" in sub_dir:
                continue
            if "flow" in sub_dir:
                continue
            sub_dir_path = os.path.join(scene_path, sub_dir, "frames")
            if not os.path.isdir(sub_dir_path):
                continue
            for img in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img)
                if not img.endswith('.png'):
                    continue
            
                with open("visionsim.txt", 'a') as f:
                    f.write(f"{img_path}\n")


def generate_file_list_for_xvfi(base_path):
    DIRS = ["val", "test"]
    for dir_name in DIRS:
        dir_path = os.path.join(base_path, dir_name)
        for sub_dir in tqdm(os.listdir(dir_path)):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            for ss_dir in os.listdir(sub_dir_path):
                ss_path = os.path.join(sub_dir_path, ss_dir)
                if not os.path.isdir(ss_path):
                    continue

                for img in os.listdir(ss_path):
                    img_path = os.path.join(ss_path, img)
                    if not img.endswith('.png'):
                        continue

                    if dir_name == "test":
                        with open("xvfi_test.txt", 'a') as f:
                            f.write(f"{img_path}\n")
                    else:
                        with open("xvfi_val.txt", 'a') as f:
                            f.write(f"{img_path}\n")

    TRAIN_DIR = os.path.join(base_path, "train")
    for dr in tqdm(os.listdir(TRAIN_DIR)):
        if not os.path.isdir(os.path.join(TRAIN_DIR, dr)):
            continue
        dr_path = os.path.join(TRAIN_DIR, dr)
        for sdr in os.listdir(dr_path):
            if not os.path.isdir(os.path.join(dr_path, sdr)):
                continue

            for img in os.listdir(os.path.join(dr_path, sdr)):
                img_path = os.path.join(dr_path, sdr, img)
                if not img.endswith('.png'):
                    continue
                with open("xvfi_train.txt", 'a') as f:
                    f.write(f"{img_path}\n")


def generate_file_list_for_div2k(base_path):
    sdirs = ["DIV2K_train_HR", "DIV2K_valid_HR"]
    for sdir in sdirs:
        sdir_path = os.path.join(base_path, sdir)
        if not os.path.isdir(sdir_path):
            continue
        for img in os.listdir(sdir_path):
            img_path = os.path.join(sdir_path, img)
            if not img.endswith('.png'):
                continue
            with open("./dataset_txt_files/div2k.txt", 'a') as f:
                f.write(f"{img_path}\n")


def generate_file_list_for_ffhq(base_path):
    imgs_dir = os.path.join(base_path, "images1024x1024")
    if not os.path.isdir(imgs_dir):
        print(f"Directory {imgs_dir} does not exist.")
        return
    for img in tqdm(os.listdir(imgs_dir)):
        img_path = os.path.join(imgs_dir, img)
        if not img.endswith('.png'):
            continue
        with open("./dataset_txt_files/ffhq.txt", 'a') as f:
            f.write(f"{img_path}\n")


def generate_file_list_for_laion170m(base_path):
    imgs_dir = os.path.join(base_path, "images")
    if not os.path.isdir(imgs_dir):
        print(f"Directory {imgs_dir} does not exist.")
        return
    for img in tqdm(os.listdir(imgs_dir)):
        img_path = os.path.join(imgs_dir, img)
        with open("./dataset_txt_files/laion.txt", 'a') as f:
            f.write(f"{img_path}\n")


def generate_file_list_for_lhq(base_path):
    dirs = ["00000-30000", "30000-60000", "60000-90000"]
    for d in dirs:
        dir_path = os.path.join(base_path, d)
        if not os.path.isdir(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue
        for img in tqdm(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img)
            if not img.endswith('.png'):
                continue
            with open("./dataset_txt_files/lhq.txt", 'a') as f:
                f.write(f"{img_path}\n")


def generate_file_list_for_REDS(base_path):
    sdirs = [os.path.join(base_path, "train"),
             os.path.join(base_path, "val")]
    for sdir in sdirs:
        if not os.path.isdir(sdir):
            print(f"Directory {sdir} does not exist.")
            continue
        for sub_dir in tqdm(os.listdir(sdir)):
            sub_dir_path = os.path.join(sdir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            for img in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img)
                if not img.endswith('.png'):
                    continue
                with open("./dataset_txt_files/reds.txt", 'a') as f:
                    f.write(f"{img_path}\n")


def generate_file_list_for_UDM10(base_path):
    for sub_dir in tqdm(os.listdir(base_path)):
        sub_dir_path = os.path.join(base_path, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for img in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img)
            if not img.endswith('.png'):
                continue
            with open("./dataset_txt_files/udm10.txt", 'a') as f:
                f.write(f"{img_path}\n")


def generate_file_list_for_Flickr2K(base_path):
    for img in tqdm(os.listdir(base_path)):
        img_path = os.path.join(base_path, img)
        if not img.endswith('.png'):
            continue
        with open("./dataset_txt_files/flickr2k.txt", 'a') as f:
            f.write(f"{img_path}\n")


def generate_file_list_for_spmc(base_path):
    for sub_dir in tqdm(os.listdir(base_path)):
        sub_dir_path = os.path.join(base_path, sub_dir, "truth")
        if not os.path.isdir(sub_dir_path):
            continue
        for img in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img)
            if not img.endswith('.png'):
                continue
            with open("./dataset_txt_files/spmc.txt", 'a') as f:
                f.write(f"{img_path}\n")


def generate_file_list_for_youhq(base_path):
    for sdir in os.path.join(base_path, "YouHQ-Extracted"):
        if not os.path.isdir(sdir):
            print(f"Directory {sdir} does not exist.")
            continue
        for sub_dir in tqdm(os.listdir(sdir)):
            sub_dir_path = os.path.join(sdir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            for img in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img)
                if not img.endswith('.png'):
                    continue
                with open("./dataset_txt_files/youhq_train.txt", 'a') as f:
                    f.write(f"{img_path}\n")

    for sdir in os.path.join(base_path, "YouHQ40-Test"):
        if not os.path.isdir(sdir):
            print(f"Directory {sdir} does not exist.")
            continue
        for img in tqdm(os.listdir(sdir)):
            img_path = os.path.join(sdir, img)
            if not img.endswith('.png'):
                continue
            with open("./dataset_txt_files/youhq_test.txt", 'a') as f:
                    f.write(f"{img_path}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate txt file lists for datasets")
    parser.add_argument('--i2', action='store_true', help='Generate txt for I2 dataset')
    parser.add_argument('--visionsim', action='store_true', help='Generate txt for VisionSim dataset')
    parser.add_argument('--xvfi', action='store_true', help='Generate txt for XVFI dataset')
    parser.add_argument('--all', action='store_true', help='Generate txt for all datasets')
    parser.add_argument('--div2k', action='store_true', help='Generate txt for DIV2K dataset')
    parser.add_argument('--ffhq', action='store_true', help='Generate txt for FFHQ dataset')
    parser.add_argument('--laion170M', action='store_true', help='Generate txt for LAION 170M dataset')
    parser.add_argument('--lhq', action='store_true', help='Generate txt for LHQ dataset')
    parser.add_argument('--reds', action='store_true', help='Generate txt for REDS dataset')
    parser.add_argument('--udm10', action='store_true', help='Generate txt for UDM10 dataset')
    parser.add_argument('--flickr2k', action='store_true', help='Generate txt for Flickr2K dataset')
    parser.add_argument('--spmc', action='store_true', help='Generate txt for SPMC dataset')
    parser.add_argument('--youhq', action='store_true', help='Generate txt for YouHQ dataset')

    parser.add_argument('--combiner', action='store_true', help='Combine all dataset txts into one')
    args = parser.parse_args()
    I2_PATH = "./i2_2000fps/extracted" 
    VISIONSIM_PATH = "./visionsim50"
    XVFI_PATH = "./xvfi"

    FFQH_PATH = "./FFHQ"
    DIV2K_PATH = "./DIV2K"
    LHQ_PATH = "./LHQ"
    REDS_PATH = "./REDS_120fps"
    UDM10_PATH = "./UDM10_video"
    FLICKR2K_PATH = "./flickr2K"
    SPMC_PATH = "./spmc_video/test_set"

    YOU_HQ_PATH = "./YouHQ"
    # TODO (Again)
    LAION170M_PATH = "./LAION-HQ"
    

    if args.all:
        args.i2 = True
        args.visionsim = True
        args.xvfi = True
    if args.i2:
        print("Generating file_list for I2 dataset...")
        generate_file_list_for_i2(base_path=I2_PATH)
    if args.visionsim:
        print("Generating file_list for VisionSim dataset...")
        generate_file_list_for_visionsim(base_path=VISIONSIM_PATH)
    if args.xvfi:
        print("Generating file_list for XVFI dataset...")
        generate_file_list_for_xvfi(base_path=XVFI_PATH)
    if args.div2k:
        print("Generating file_list for DIV2K dataset...")
        generate_file_list_for_div2k(base_path=DIV2K_PATH)
    if args.ffhq:
        print("Generating file_list for FFHQ dataset...")
        generate_file_list_for_ffhq(base_path=FFQH_PATH)
    if args.laion170M:
        print("Generating file_list for LAION 170M dataset...")
        generate_file_list_for_laion170m(base_path=LAION170M_PATH)
    if args.lhq:
        print("Generating file_list for LHQ dataset...")
        generate_file_list_for_lhq(base_path=LHQ_PATH)
    if args.reds:
        print("Generating file_list for REDS dataset...")
        generate_file_list_for_REDS(base_path=REDS_PATH)
    if args.udm10:
        print("Generating file_list for UDM10 dataset...")
        generate_file_list_for_UDM10(base_path=UDM10_PATH)
    if args.flickr2k:
        print("Generating file_list for Flickr2K dataset...")
        generate_file_list_for_Flickr2K(base_path=FLICKR2K_PATH)
    if args.spmc:
        print("Generating file_list for SPMC dataset...")
        generate_file_list_for_spmc(base_path=SPMC_PATH)
    if args.youhq:
        print("Generating file_list for YouHQ dataset...")
        generate_file_list_for_youhq(base_path=YOU_HQ_PATH)


    if args.combiner:
        print("Combining all dataset txts into one...")
        with open("./dataset_txt_files/combined_dataset.txt", 'w') as combined_file:
            for dataset in ["i2k2_train.txt", "i2k2_test.txt", 
                            "visionsim.txt", "laion.txt",
                            "youhq_train.txt", "youhq_test.txt",
                            "xvfi_train.txt", "xvfi_val.txt", "xvfi_test.txt", 
                            "div2k.txt", "ffhq.txt", "reds.txt", "udm10.txt", "flickr2k.txt", "spmc.txt", "lhq.txt"]:
                ds = os.path.join("./dataset_txt_files", dataset)
                if os.path.exists(ds):
                    with open(ds, 'r') as f:
                        lines = f.readlines()
                        lines = sorted(lines)
                        combined_file.writelines(lines)     
        print("Combined file list created: combined_dataset.txt")

