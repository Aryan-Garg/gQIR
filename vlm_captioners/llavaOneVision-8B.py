#!/usr/bin/env python3
# Code derived from: https://huggingface.co/docs/transformers/en/model_doc/llava_onevision#llava-onevision
import math
import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-si-hf")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-si-hf",
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "Please describe the image in detail in a single paragraph."}
                    ],
                },
            ]
Prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

I2_PATH = "../i2_2000fps/extracted" 
VISIONSIM_PATH = "../visionsim50"
XVFI_PATH = "../xvfi"

def generate_prompts_for_i2(base_path=I2_PATH):
    for sub_dir in tqdm(os.listdir(base_path)):
        if os.path.exists(base_path, f'{sub_dir}.json'):
            continue
        sub_dir_path = os.path.join(base_path, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for img in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img)
            if not img.endswith('.png'):
                continue
            pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16)
            inputs = processor(images=pixel_values, text=Prompt, return_tensors="pt").to("cuda:0")
            outputs = model.generate(**inputs)
            response = processor.decode(outputs[0], skip_special_tokens=True)
            parsed_response = response.split("\n")[-1]
            # append this {img_path: response} to a json file instantly instead of dumping all at once
            with open(os.path.join(base_path, f'{sub_dir}.json'), 'a') as f:
                json.dump({img_path: parsed_response}, f, indent=4)


def generate_prompts_for_visionsim(base_path=VISIONSIM_PATH):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    for scene_name in tqdm(os.listdir(base_path)):
        scene_path = os.path.join(base_path, scene_name)
        if not os.path.isdir(scene_path):
            continue
        for sub_dir in os.listdir(scene_path):
            if "frames" not in sub_dir and "spc" in sub_dir:
                continue
            if "flow" in sub_dir:
                continue
            if "4x" in sub_dir:
                continue
            if os.path.exists(os.path.join(scene_path, f'{sub_dir}.json')):
                continue
            sub_dir_path = os.path.join(scene_path, sub_dir, "frames")
            if not os.path.isdir(sub_dir_path):
                continue
            for img in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img)
                if not img.endswith('.png'):
                    continue
                pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                inputs = processor(images=pixel_values, text=Prompt, return_tensors="pt").to("cuda:0")
                outputs = model.generate(**inputs)
                response = processor.decode(outputs[0], skip_special_tokens=True)
                parsed_response = response.split("\n")[-1]
                with open(os.path.join(scene_path, f'{sub_dir}.json'), 'a') as f:
                    json.dump({img_path: parsed_response}, f, indent=4)


def generate_prompts_for_xvfi(base_path=XVFI_PATH):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    DIRS = ["val", "test"]
    for dir_name in DIRS:
        dir_path = os.path.join(base_path, dir_name)
        for sub_dir in tqdm(os.listdir(dir_path)):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            for ss_dir in os.listdir(sub_dir_path):
                if os.path.exists(os.path.join(sub_dir_path, f'{ss_dir}.json')):
                    continue
                if not os.path.isdir(os.path.join(sub_dir_path, ss_dir)):
                    continue
                ss_path = os.path.join(sub_dir_path, ss_dir)
                if not os.path.isdir(ss_path):
                    continue

                for img in os.listdir(ss_path):
                    img_path = os.path.join(ss_path, img)
                    if not img.endswith('.png'):
                        continue
                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    inputs = processor(images=pixel_values, text=Prompt, return_tensors="pt").to("cuda:0")
                    outputs = model.generate(**inputs)
                    response = processor.decode(outputs[0], skip_special_tokens=True)
                    parsed_response = response.split("\n")[-1]

                    with open(os.path.join(sub_dir_path, f'{ss_dir}.json'), 'a') as f:
                        json.dump({img_path: parsed_response}, f, indent=4)
    
    TRAIN_DIR = os.path.join(base_path, "train")
    for dr in tqdm(os.listdir(TRAIN_DIR)):
        if not os.path.isdir(os.path.join(TRAIN_DIR, dr)):
            continue
        dr_path = os.path.join(TRAIN_DIR, dr)
        for sdr in os.listdir(dr_path):
            if os.path.exists(os.path.join(dr_path, f'{sdr}.json')):
                continue
            if not os.path.isdir(os.path.join(dr_path, sdr)):
                continue

            for img in os.listdir(os.path.join(dr_path, sdr)):
                img_path = os.path.join(dr_path, sdr, img)
                if not img.endswith('.png'):
                    continue
                pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                inputs = processor(images=pixel_values, text=Prompt, return_tensors="pt").to("cuda:0")
                outputs = model.generate(**inputs)
                response = processor.decode(outputs[0], skip_special_tokens=True)
                parsed_response = response.split("\n")[-1]
        
                with open(os.path.join(dr_path, f'{sdr}.json'), 'a') as f:
                    json.dump({img_path: parsed_response}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for InternVL3-8B")
    parser.add_argument('--i2', action='store_true', help='Generate prompts for I2 dataset')
    parser.add_argument('--visionsim', action='store_true', help='Generate prompts for VisionSim dataset')
    parser.add_argument('--xvfi', action='store_true', help='Generate prompts for XVFI dataset')
    args = parser.parse_args()
    if args.i2:
        print("Generating prompts for I2 dataset...")
        generate_prompts_for_i2(base_path=I2_PATH)
    if args.visionsim:
        print("Generating prompts for VisionSim dataset...")
        generate_prompts_for_visionsim(base_path=VISIONSIM_PATH)
    if args.xvfi:
        print("Generating prompts for XVFI dataset...")
        generate_prompts_for_xvfi(base_path=XVFI_PATH)
