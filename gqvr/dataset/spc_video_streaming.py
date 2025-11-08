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
from torch.utils.data import get_worker_info


class StreamingSlidingVideoDataset(IterableDataset):
    def __init__(self,
                    file_list: str,
                    file_backend_cfg: Mapping[str, Any],
                    out_size: int,
                    crop_type: str,
                    use_hflip: bool,
                    sliding_window: int = 1 ,
                    chunk_size: int = 11,
                    mosaic: bool = False) -> "StreamingSlidingVideoDataset":
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
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.bits = 3
        self.mosaic = mosaic
        print(f"[+] Sim bits = {self.bits}")


    def _load_video(self, video_path, rng=None):
        if video_path.startswith("./"):
            correct_video_path =  self.HARDDISK_DIR + video_path[2:]
        else:
            correct_video_path =  video_path
        png_files = sorted([f for f in os.listdir(correct_video_path) if f.endswith(".png") or f.endswith(".jpg")])
        gts = []
        lqs = []
        for img_name in png_files:
            image_path = os.path.join(correct_video_path, img_name)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.out_size, self.out_size), Image.LANCZOS)
            image = np.array(image)
            # print(f"Loaded GT image size: {image.size}")
            # if self.crop_type != "none":
            #     if image.height == self.out_size and image.width == self.out_size:
            #         image = np.array(image)
            #     else:
            #         if self.crop_type == "center":
            #             image = center_crop_arr(image, self.out_size)
            #         elif self.crop_type == "random":
            #             image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
            # else:
            #     assert image.height == self.out_size and image.width == self.out_size
            #     image = np.array(image)
                # hwc, rgb, 0,255, uint8

            img_lq_sum = np.zeros_like(image, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for k in range(N): # 3-bit (2**3 - 1)
                if self.mosaic:
                    bayer_pattern_type = rng.choice(["RGGB","GRBG","BGGR","GBRG"])
                    img_lq_sum = img_lq_sum + self.get_mosaic(self.generate_spc_from_gt(image), bayer_pattern_type)
                else:
                    img_lq_sum = img_lq_sum + self.generate_spc_from_gt(image)
            img_lq = img_lq_sum / (1.0*N)
            lqs.append(img_lq)
            gts.append(image)


        return np.stack(gts, axis=0), np.stack(lqs, axis=0)


    def get_mosaic(self, img, bayer_pattern_type):
        """
            Convert a demosaiced RGB image (HxWx3) into an RGGB Bayer mosaic.
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        bayer = np.zeros_like(img)

        # bayer_pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])

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


    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            vid_iter = enumerate(self.video_files)
        else:
            # split video list across workers
            per_worker = (len(self.video_files) + worker.num_workers - 1) // worker.num_workers
            start = worker.id * per_worker
            end = min(start + per_worker, len(self.video_files))
            vid_iter = enumerate(self.video_files[start:end], start)

        rng = np.random.RandomState(42 + (worker.id if worker else 0))  # deterministic per worker


        for _, video_info in vid_iter:
            video_path = video_info["video_path"]
            # print(f"Loading video from {video_path}")
            try:
                gts, lqs = self._load_video(video_path, rng=rng)
            except Exception as e:
                print(f"Error loading video from {video_path}: {e}")
                continue

            gts = (gts / 255.0).astype(np.float32)
            gts = (gts * 2) - 1
            lqs = ((lqs*2) - 1).astype(np.float32)
            
            # Sliding window
            chunk_size = min(self.chunk_size, gts.shape[0])
            T_total = gts.shape[0]
            for start_idx in range(0, T_total -  chunk_size + 1, self.sliding_window):
                gt_chunk = gts[start_idx:start_idx + chunk_size] 
                lq_chunk = lqs[start_idx:start_idx + chunk_size]
                yield {"gts": gt_chunk, "lqs": lq_chunk}
    

if __name__ == "__main__":
    dataset = StreamingSlidingVideoDataset(
        file_list="/media/agarg54/Extreme SSD/video_dataset_txt_files/udm10.txt",
        file_backend_cfg={"target": "gqvr.dataset.file_backend.HardDiskBackend"},
        out_size=512,
        crop_type="center",
        use_hflip=False,
        sliding_window=1,
        chunk_size=11
    )
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(dataloader):
        print(i)
        print(batch["gts"].shape)
        print(batch["lqs"].shape)