import os
import json
from tqdm import tqdm

I2_PATH = "../i2_2000fps/extracted" 
VISIONSIM_PATH = "../visionsim50"
XVFI_PATH = "../xvfi"

def generate_prompts_for_i2(base_path=I2_PATH):
    for sub_dir in tqdm(os.listdir(base_path)):
        sub_dir_path = os.path.join(base_path, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        prompts_dict = {}
        for img in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img)
            if not img.endswith('.png'):
                continue
            response = ""
            prompts_dict[img_path] = response
        with open(os.path.join(base_path, f'null_{sub_dir}.json'), 'w') as f:
            json.dump(prompts_dict, f, indent=4)


def generate_prompts_for_visionsim(base_path=VISIONSIM_PATH):
    for scene_name in tqdm(sorted(os.listdir(base_path))):
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
            prompts_dict = {}
            for img in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img)
                if not img.endswith('.png'):
                    continue
                response = ""
                prompts_dict[img_path] = response
            with open(os.path.join(scene_path, f'null_{sub_dir}.json'), 'w') as f:
                json.dump(prompts_dict, f, indent=4)


def generate_prompts_for_xvfi(base_path=XVFI_PATH):
    DIRS = ["val", "test"]
    for dir_name in DIRS:
        dir_path = os.path.join(base_path, dir_name)
        for sub_dir in tqdm(os.listdir(dir_path)):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            for ss_dir in os.listdir(sub_dir_path):
                ss_path = os.path.join(sub_dir_path, ss_dir)
                if not os.path.isdir(ss_path):
                    continue
                prompts_dict = {}
                for img in os.listdir(ss_path):
                    img_path = os.path.join(ss_path, img)
                    if not img.endswith('.png'):
                        continue
                    response = ""
                    prompts_dict[img_path] = response
                with open(os.path.join(sub_dir_path, f'null_{ss_dir}.json'), 'w') as f:
                    json.dump(prompts_dict, f, indent=4)
    
    TRAIN_DIR = os.path.join(base_path, "train")
    for dr in tqdm(os.listdir(TRAIN_DIR)):
        if not os.path.isdir(os.path.join(TRAIN_DIR, dr)):
            continue
        dr_path = os.path.join(TRAIN_DIR, dr)
        for sdr in os.listdir(dr_path):
            if not os.path.isdir(os.path.join(dr_path, sdr)):
                continue
            prompts_dict = {}
            for img in os.listdir(os.path.join(dr_path, sdr)):
                img_path = os.path.join(dr_path, sdr, img)
                if not img.endswith('.png'):
                    continue
                response = ""
                prompts_dict[img_path] = response
            with open(os.path.join(dr_path, f'null_{sdr}.json'), 'w') as f:
                json.dump(prompts_dict, f, indent=4)


print("Generating prompts for I2 dataset...")
generate_prompts_for_i2(base_path=I2_PATH)
print("Generating prompts for VisionSim dataset...")
generate_prompts_for_visionsim(base_path=VISIONSIM_PATH)
print("Generating prompts for XVFI dataset...")
generate_prompts_for_xvfi(base_path=XVFI_PATH)