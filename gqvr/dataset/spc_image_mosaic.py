from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import importlib

import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image
import torch.utils.data as data

from .utils import load_file_list, center_crop_arr, random_crop_arr, srgb_to_linearrgb, emulate_spc
from ..utils.common import instantiate_from_config


class SPCDataset_Mosaic(data.Dataset):
    """
        Dataset for finetuning the VAE's encoder and Adversarial FT Stages (independent of each other).
        Args:
            file_list (str): Path to the file list containing image paths and prompts.
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
                    use_hflip: bool,
                    bits=3) -> "SPCDataset_Mosaic":

        super(SPCDataset_Mosaic, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip # No need for 1.5M big dataset
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.bits = bits
        print(f"[+] Sim bits = {self.bits}")


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


    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        # TODO: Add linearrgb to spad curve 
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        img_lq = None
        while img_gt is None and img_lq is None:
            # load meta file
            img_path = self.image_files[index]['image_path']
            gt_path =  self.HARDDISK_DIR + img_path[2:]
            # print("gt path:", gt_path)
            # print(f"Loading GT image from {gt_path}")
            prompt = self.image_files[index]['prompt']

            try:
                img_gt = self.load_gt_image(gt_path)
            except Exception as e:
                print(e)
                print(f"Could not load: {gt_path}, setting a random index")
                index = random.randint(0, len(self) - 1)
                continue
            
            if img_gt is None:
                print(f"failed to load {gt_path} or generate lq image, try another image")
                index = random.randint(0, len(self) - 1)
                continue


            img_lq_sum = np.zeros_like(img_gt, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for i in range(N): # 3-bit (2**3 - 1)
                img_lq_sum = img_lq_sum + self.get_mosaic(self.generate_spc_from_gt(img_gt))
            img_lq = img_lq_sum / (1.0*N)


        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = img_lq.astype(np.float32) # BUG-FIXED now!!! for all datasets img_lq is already [0,1], no need to divide by 255

        # if self.use_hflip and np.random.uniform() < 0.5:
        #     img_gt = np.fliplr(img_gt)
        #     img_lq = np.fliplr(img_lq)

        # Should lq be normalized to [-1,1] or stay in [0, 1] range? For now [-1, 1]
        gt = (img_gt * 2 - 1).astype(np.float32)
        # [-1, 1]
        lq = (img_lq * 2 - 1).astype(np.float32) 
        # print(np.amax(lq), np.amin(lq))
        return gt, lq, prompt, gt_path


    def __len__(self) -> int:
        return len(self.image_files)
    

if __name__ == "__main__":
    # Testing/Example usage
    dataset = SPCDataset_Mosaic(
        file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/combined_dataset.txt",
        file_backend_cfg={"target": "core.dataset.file_backend.HardDiskBackend"},
        out_size=512,
        crop_type="center",
        use_hflip=False,
        bits = 3
    )
    print(f"Complete Dataset length: {len(dataset)}")
    sample = next(iter(dataset))
    print(f"Sample GT shape: {sample[0].shape}, LQ shape: {sample[1].shape}, Prompt: {sample[2]}")
    print(f"{np.amax(sample[1])} | {np.amin(sample[1])}")
    import matplotlib.pyplot as plt
    plt.imsave("GT.png", (sample[0] - np.amin(sample[0])) / (np.amax(sample[0]) - np.amin(sample[0])))
    plt.imsave("SPC.png", (sample[1] - np.amin(sample[1])) / (np.amax(sample[1]) - np.amin(sample[1])))