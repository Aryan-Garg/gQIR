from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import importlib
import os

import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image
import torch.utils.data as data

from .utils import load_video_file_list, center_crop_arr, random_crop_arr, srgb_to_linearrgb, emulate_spc
from ..utils.common import instantiate_from_config


class SPCVideoDataset(data.Dataset):
    """
        Dataset for finetuning the VAE's encoder and SPC-ControlNet Stages (independent of each other).
        Args:
            file_list (str): Path to the file list containing image paths and optionally prompts.
            file_backend_cfg (Mapping[str, Any]): Configuration for the file backend to load images.
            out_size (int): The output size of the images after cropping.
            crop_type (str): Type of cropping to apply to the images. Options are 'none', 'center', or 'random'.
        Returns:
            A dictionary containing:
                - 'gt': Ground truth image tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'lq': SPC image (dubbed as low-quality) tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'prompt': The prompt associated with the image.
    """
    def __init__(self,
                    file_list: str,
                    file_backend_cfg: Mapping[str, Any],
                    out_size: int,
                    crop_type: str,
                    use_hflip: bool) -> "SPCVideoDataset":

        super(SPCVideoDataset, self).__init__()
        self.file_list = file_list
        self.video_files = load_video_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip # No need for 1.5M big dataset
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/mnt/disks/behemoth/datasets/"


    def load_gt_images(self, video_path: str, max_retry: int = 5):
        gt_images = []
        # print(f"Loading GT video from {video_path}")
        for img_name in sorted(os.listdir(video_path)):
            image_path = os.path.join(video_path, img_name)
            # print(f"Loading {image_path}")
            image = Image.open(image_path).convert("RGB")
            # print(f"Loaded GT image size: {image.size}")
            if self.crop_type != "none":
                if image.height == self.out_size and image.width == self.out_size:
                    image = np.array(image)
                else:
                    if self.crop_type == "center":
                        image = center_crop_arr(image, self.out_size)
                    elif self.crop_type == "random":
                        image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
            else:
                assert image.height == self.out_size and image.width == self.out_size
                image = np.array(image)
            # hwc, rgb, 0,255, uint8
            gt_images.append(image)
        return np.stack(gt_images, axis=0) # thwc


    def generate_spc_from_gt(self, img_gt):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        img = emulate_spc(img, 
                          factor=1.0 # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img


    def convert_to_Nbit_spc(self, imgs_gt: npt.NDArray, bits: int = 3):
        N = 2**bits - 1
        imgs_lq = []
        for img_gt in imgs_gt:
            img_lq_sum = np.zeros_like(img_gt, dtype=np.float32)
            for i in range(N): # 4-bit (2**4 - 1)
                img_lq_sum = img_lq_sum + self.generate_spc_from_gt(img_gt)
            img_lq = img_lq_sum / (1.0*N)
            imgs_lq.append(img_lq)
        return np.stack(imgs_lq, axis=0) # thwc


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        imgs_gt = None
        imgs_lq = None
        while imgs_gt is None and imgs_lq is None:
            # load meta file
            video_path = self.video_files[index]['video_path']
            gt_video_path =  self.HARDDISK_DIR + video_path[2:]
            # print("gt path:", gt_path)
            # print(f"Loading GT image from {gt_path}")
            prompt = self.video_files[index]['prompt']

            try:
                imgs_gt = self.load_gt_images(gt_video_path)
                # print(f"Loaded {imgs_gt.shape[0]} frames from {gt_video_path}")
            except Exception as e:
                print(e)
                print(f"Could not load: {gt_video_path}, setting a random index")
                index = random.randint(0, len(self) - 1)
                continue
            
            if imgs_gt is None:
                print(f"failed to load {gt_video_path} or generate lq image, try another image")
                index = random.randint(0, len(self) - 1)
                continue

            # NOTE: SPAD bit-resolution was changed permanently --- No need for 1-bit VAEs
            # However, to revert back... uncomment:
            # img_lq = self.generate_spc_from_gt(img_gt)
            # And comment the following
            
            spc_bits = 3 # bits
            imgs_lq = self.convert_to_Nbit_spc(imgs_gt, bits=spc_bits)


        # Shape: (t, h, w, c); channel order: RGB; image range: [0, 1], float32.
        imgs_gt = (imgs_gt / 255.0).astype(np.float32)
        imgs_lq = imgs_lq.astype(np.float32) # BUG-FIXED now!!! for all datasets img_lq is already [0,1], no need to divide by 255

        # if self.use_hflip and np.random.uniform() < 0.5:
        #     img_gt = np.fliplr(img_gt)
        #     img_lq = np.fliplr(img_lq)

        # Should lq be normalized to [-1,1] or stay in [0, 1] range? For now [-1, 1]
        gt = (imgs_gt * 2 - 1).astype(np.float32)
        # [-1, 1]
        lq = (imgs_lq * 2 - 1).astype(np.float32) 
        # print(np.amax(lq), np.amin(lq))
        return gt, lq, prompt, gt_video_path


    def __len__(self) -> int:
        return len(self.video_files)
    

if __name__ == "__main__":
    # Testing/Example usage
    dataset = SPCVideoDataset(
        file_list="/home/argar/apgi/gQVR/dataset_txt_files/udm10_video.txt",
        file_backend_cfg={"target": "gqvr.dataset.file_backend.HardDiskBackend"},
        out_size=512,
        crop_type="center",
        use_hflip=False,
    )
    print(f"Complete Dataset length: {len(dataset)}")
    sample = next(iter(dataset))
    print(f"Sample GT shape: {sample[0].shape}, LQ shape: {sample[1].shape}, Prompt: {sample[2]}, Video Path: {sample[3]}")
    print(f"GT Range: {np.amax(sample[0])} | {np.amin(sample[0])}")
    print(f"SPC Range: {np.amax(sample[1])} | {np.amin(sample[1])}")
    # import matplotlib.pyplot as plt
    # plt.imsave("GT.png", (sample[0] - np.amin(sample[0])) / (np.amax(sample[0]) - np.amin(sample[0])))
    # plt.imsave("SPC.png", (sample[1] - np.amin(sample[1])) / (np.amax(sample[1]) - np.amin(sample[1])))