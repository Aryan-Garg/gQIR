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
import torch
import torch.utils.data as data
from torch.utils.data import IterableDataset

from .utils import load_video_file_list, center_crop_arr, random_crop_arr, srgb_to_linearrgb, emulate_spc
from ..utils.common import instantiate_from_config


class SlidingLatentVideoDataset(IterableDataset):
    """
    Streams multiple precomputed latent videos from disk and yields sliding windows of T frames.
    Assumes precomputed latents are stored as .pt tensors.
    """
    def __init__(self,
                    file_list: str,
                    file_backend_cfg: Mapping[str, Any],
                    out_size: int,
                    crop_type: str,
                    use_hflip: bool,
                    precomputed_latents: bool, 
                    sliding_window: int,
                    chunk_size: int) -> "SlidingLatentVideoDataset":
        """
        Args:
            video_files (list of dict): each dict must contain:
                - 'video_path': str, path to video folder with .pt files
                - 'prompt': str, associated prompt
            chunk_size (int): number of frames per chunk (T)
            device (str): device to load tensors onto
        """
        self.file_list = file_list
        self.video_files = load_video_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip # No need for 1.5M big dataset
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/mnt/disks/behemoth/datasets/"
        self.precomputed_latents = precomputed_latents
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size


    def _load_video(self, video_path):
        """Load all precomputed latent tensors from a single video folder."""
        correct_video_path =  self.HARDDISK_DIR + video_path[2:]
        latent_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".pt")])
        latents = []
        for lf in latent_files:
            latent = torch.load(os.path.join(correct_video_path, lf), map_location="cpu")  # [1, 4 , 64, 64]
            latents.append(latent)

        png_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".png")])
        gts = []
        for img_name in png_files:
            image_path = os.path.join(correct_video_path, img_name)
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
            gts.append(image)

        if len(latents) == 0:
            return None
        
        return torch.cat(latents, dim=0) , np.stack(gts, axis=0) # [T_total, 4, 64, 64]; [T_total, H, W, C]


    def __iter__(self):
        for video_info in self.video_files:
            video_path = video_info["video_path"]

            latents, gts = self._load_video(video_path)
            if latents is None or len(latents) < self.chunk_size:
                continue

            # Sliding window over latents
            T_total = latents.shape[0]
            for start_idx in range(0, T_total - self.chunk_size + 1, self.sliding_window): # Currently manually set in the init fn
                chunk = latents[start_idx:start_idx + self.chunk_size]  # [T, C, H, W]
                gt_chunk = gts[start_idx:start_idx + self.chunk_size] 
                yield {
                    "latents": chunk,        # [T, C, H, W]
                    "gts": gt_chunk,
                    "video_path": video_path
                }
    

# if __name__ == "__main__":
#     # Testing/Example usage
#     dataset = SPCVideoDataset(
#         file_list="/home/argar/apgi/gQVR/dataset_txt_files/udm10_video.txt",
#         file_backend_cfg={"target": "gqvr.dataset.file_backend.HardDiskBackend"},
#         out_size=512,
#         crop_type="center",
#         use_hflip=False,
#     )
#     print(f"Complete Dataset length: {len(dataset)}")
#     sample = next(iter(dataset))
#     print(f"Sample GT shape: {sample[0].shape}, LQ shape: {sample[1].shape}, Prompt: {sample[2]}, Video Path: {sample[3]}")
#     print(f"GT Range: {np.amax(sample[0])} | {np.amin(sample[0])}")
#     print(f"SPC Range: {np.amax(sample[1])} | {np.amin(sample[1])}")
#     # import matplotlib.pyplot as plt
    # plt.imsave("GT.png", (sample[0] - np.amin(sample[0])) / (np.amax(sample[0]) - np.amin(sample[0])))
    # plt.imsave("SPC.png", (sample[1] - np.amin(sample[1])) / (np.amax(sample[1]) - np.amin(sample[1])))