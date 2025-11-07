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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data


from .utils import load_video_file_list, center_crop_arr, random_crop_arr, srgb_to_linearrgb, emulate_spc
from ..utils.common import instantiate_from_config


class VideoDataset(data.Dataset):
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
                    sliding_window: int,
                    chunk_size: int,
                    precomputed_latents: bool = False,
                    mosaic: bool = False) -> "VideoDataset":
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
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.precomputed_latents = precomputed_latents
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.bits = 3
        self.mosaic = mosaic
        print(f"[+] Mosaic: {self.mosaic}")
        print(f"[+] Sim bits = {self.bits}")


    def __len__(self):
        return len(self.video_files)

    def _load_video(self, video_path, chunk_size):
        """Load all precomputed latent tensors from a single video folder."""
        correct_video_path =  self.HARDDISK_DIR + video_path[2:]
        if self.precomputed_latents:
            latent_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".pt")])
            latents = []
            for lf in latent_files:
                latent = torch.load(os.path.join(correct_video_path, lf), map_location="cpu")  # [1, 4 , 64, 64]
                latents.append(latent)

        png_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".png")])
        T_total = len(png_files)
        start_idx = random.randint(0, max(0, T_total - chunk_size)) if T_total > chunk_size else 0
        gts = []
        lqs = []
        for i in range(start_idx, start_idx + chunk_size):
            img_name = png_files[i]
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

            img_lq_sum = np.zeros_like(image, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for i in range(N): # 3-bit (2**3 - 1)
                if self.mosaic:
                    img_lq_sum = img_lq_sum + self.get_mosaic(self.generate_spc_from_gt(image))
                else:
                    img_lq_sum = img_lq_sum + self.generate_spc_from_gt(image)
            img_lq = img_lq_sum / (1.0*N)
            lqs.append(img_lq)
            gts.append(image)

        if self.precomputed_latents:     
            return torch.cat(latents, dim=0) , np.stack(gts, axis=0), np.stack(lqs, axis=0) # [T_total, 4, 64, 64]; [T_total, H, W, C]
        else:
            return None, np.stack(gts, axis=0), np.stack(lqs, axis=0)


    def get_mosaic(self, img):
        """
            Convert a demosaiced RGB image (HxWx3) into an RGGB Bayer mosaic.
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        bayer = np.zeros_like(img)

        bayer_pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])

        if bayer_pattern_type == "RGGB":
            # Red
            bayer[0::2, 0::2, 0] = R[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Blue
            bayer[1::2, 1::2, 2] = B[1::2, 1::2]
        elif bayer_pattern_type == "GRBG":
            # Red
            bayer[0::2, 1::2, 0] = R[0::2, 1::2]
            # Green 
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[1::2, 0::2, 2] = B[1::2, 0::2]
            
        elif bayer_pattern_type == "BGGR":
            # Blue
            bayer[0::2, 0::2, 2] = B[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Red
            bayer[1::2, 1::2, 0] = R[1::2, 1::2]
        
        else: # GBRG
            # Green
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[0::2, 1::2, 2] = B[0::2, 1::2]
            # Red
            bayer[1::2, 0::2, 0] = R[1::2, 0::2]

        return bayer

    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        # TODO: Add linearrgb to spad curve 
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img

    def __getitem__(self, idx):
        while True:
            try:
                video_info = self.video_files[idx]
                video_path = video_info["video_path"]
                # print(f"Loading video from {video_path}")
                latents, gts, lqs = self._load_video(video_path, self.chunk_size)
                break
            except Exception as e:
                print(f"Error loading video from {video_path}: {e}")
                # reset idx
                idx = random.randint(0, len(self.video_files) - 1)

        gts = (gts / 255.0).astype(np.float32)
        gts = (gts * 2) - 1
        lqs = ((lqs*2) - 1).astype(np.float32)
    
        
        if self.precomputed_latents:
            return {"latents": latents,        # [T, C, H, W]
                    "gts": gts, 
                    "lqs": lqs}  # [T, H, W, C], [-1, 1], float32
        else:
            return {"gts": gts, "lqs": lqs}
    


class VideoDatasetSlidinggggggggg(data.Dataset):
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
                    sliding_window: int,
                    chunk_size: int,
                    precomputed_latents: bool = False,
                    mosaic: bool = False) -> "VideoDatasetSlidinggggggggg":
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
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.precomputed_latents = precomputed_latents
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.bits = 3
        self.mosaic = mosaic
        print(f"[+] Mosaic: {self.mosaic}")
        print(f"[+] Sim bits = {self.bits}")


    def __len__(self):
        return len(self.video_files)

    def _load_video(self, correct_video_path, chunk_size, start_idx=0):
        """Load all precomputed latent tensors from a single video folder."""
        png_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".png")])
        gts = []
        lqs = []
        for i in range(start_idx, start_idx + chunk_size):
            img_name = png_files[i]
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

            img_lq_sum = np.zeros_like(image, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for i in range(N): # 3-bit (2**3 - 1)
                if self.mosaic:
                    img_lq_sum = img_lq_sum + self.get_mosaic(self.generate_spc_from_gt(image))
                else:
                    img_lq_sum = img_lq_sum + self.generate_spc_from_gt(image)
            img_lq = img_lq_sum / (1.0*N)
            lqs.append(img_lq)
            gts.append(image)

       

        return None, np.stack(gts, axis=0), np.stack(lqs, axis=0)


    def get_mosaic(self, img):
        """
            Convert a demosaiced RGB image (HxWx3) into an RGGB Bayer mosaic.
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        bayer = np.zeros_like(img)

        bayer_pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])

        if bayer_pattern_type == "RGGB":
            # Red
            bayer[0::2, 0::2, 0] = R[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Blue
            bayer[1::2, 1::2, 2] = B[1::2, 1::2]
        elif bayer_pattern_type == "GRBG":
            # Red
            bayer[0::2, 1::2, 0] = R[0::2, 1::2]
            # Green 
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[1::2, 0::2, 2] = B[1::2, 0::2]
            
        elif bayer_pattern_type == "BGGR":
            # Blue
            bayer[0::2, 0::2, 2] = B[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Red
            bayer[1::2, 1::2, 0] = R[1::2, 1::2]
        
        else: # GBRG
            # Green
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[0::2, 1::2, 2] = B[0::2, 1::2]
            # Red
            bayer[1::2, 0::2, 0] = R[1::2, 0::2]

        return bayer

    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        # TODO: Add linearrgb to spad curve 
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img

    def __getitem__(self, idx):
        while True:
            try:
                video_info = self.video_files[idx]
                video_path = video_info["video_path"]
                correct_video_path =  self.HARDDISK_DIR + video_path[2:]
                png_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".png")])
                T_total = len(png_files)
                last_idx = max(0, T_total - self.sliding_window)
                start_idx = 0
                for s_idx in range(0, last_idx + 1, self.sliding_window):
                    start_idx = s_idx
                    latents, gts, lqs = self._load_video(correct_video_path, self.chunk_size, start_idx=start_idx)
                    gts = (gts / 255.0).astype(np.float32)
                    gts = (gts * 2) - 1
                    lqs = ((lqs*2) - 1).astype(np.float32)
                    return {"gts": gts, "lqs": lqs}
                break
            except Exception as e:
                print(f"Error loading video from {video_path}: {e}")
                # reset idx
                idx = random.randint(0, len(self.video_files) - 1)



if __name__ == "__main__":
    dataset = VideoDataset(
        file_list="/media/agarg54/Extreme SSD/video_dataset_txt_files/udm10.txt",
        file_backend_cfg={"target": "gqvr.dataset.file_backend.HardDiskBackend"},
        out_size=512,
        crop_type="center",
        use_hflip=False,
        precomputed_latents=False,
        sliding_window=28,
        chunk_size=30
    )
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(dataloader):
        print(i)
        print(batch["gts"].shape)
        print(batch["lqs"].shape)