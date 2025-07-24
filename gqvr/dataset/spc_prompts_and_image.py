from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import importlib

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class SPC_Prompts_Dataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str) -> "SPC_Prompts_Dataset":
        super(SPC_Prompts_Dataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/mnt/disks/behemoth/datasets/"

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        img_lq = None
        while img_gt is None and img_lq is None:
            # load meta file
            gt_path = self.image_files[index]['image_path']
            gt_path =  self.HARDDISK_DIR + gt_path[2:]
            # print(f"Loading GT image from {gt_path}")
            prompt = self.image_files[index]['prompt']

            img_gt = self.load_gt_image(gt_path)
            if "xvfi" in gt_path:
                path_elements = gt_path.split("/")
                lq_path = f"{path_elements[0]}/{path_elements[1]}/{path_elements[2]}_{path_elements[3]}_{path_elements[4]}/{path_elements[5]}"
            elif "i2_2000fps" in gt_path:
                lq_path = gt_path.replace("extracted", "spc")
            elif "visionsim" in gt_path:
                if "frames_1x" in gt_path:
                    lq_path = gt_path.replace("frames_1x/frames", "spc_frames_1x")
                elif "frames_4x" in gt_path:
                    lq_path = gt_path.replace("frames_4x/frames", "spc_frames_4x")
             

            img_lq = self.load_gt_image(lq_path)
            if img_gt is None or img_lq is None:
                print(f"filed to load {gt_path} or {lq_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = (img_lq / 255.0).astype(np.float32)

        if np.random.uniform() < 0.5: # cfg
            prompt = ""

        # [-1, 1]
        gt = (img_gt * 2 - 1).astype(np.float32)
        # [-1, 1]
        lq = (img_lq * 2 - 1).astype(np.float32) 
        # Should lq be normalized to [-1,1] or stay in [0, 1] range? 

        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
    

if __name__ == "__main__":
    # Testing/Example usage
    dataset = SPC_Prompts_Dataset(
        file_list="/home/argar/apgi/gQVR/dataset_txt_files/combined_dataset.txt",
        file_backend_cfg={"target": "gqvr.dataset.file_backend.HardDiskBackend"},
        out_size=512,
        crop_type="center",
    )
    print(f"Complete Dataset length: {len(dataset)}")
    sample = next(iter(dataset))
    print(f"Sample GT shape: {sample[0].shape}, LQ shape: {sample[1].shape}, Prompt: {sample[2]}")