"""Data loading utilities for Quanta Burst Photography."""

from __future__ import annotations

import json
import warnings
from math import floor
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from einops import rearrange, reduce
from loguru import logger
from natsort import natsorted
from scipy import ndimage, spatial
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

BayerPatternLiteral = Literal["gray", "grey", "rggb", "bggr", "grbg", "gbrg"]


def nearest_neighbor_indices(
    bad_pixel_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nearest valid neighbor indices for bad pixels (grayscale)."""
    query_i, query_j = np.where(bad_pixel_mask.astype(bool))
    source_i, source_j = np.where(~bad_pixel_mask.astype(bool))
    if source_i.size == 0:
        logger.warning("No valid pixels found for hot-pixel correction.")
        return query_i, query_j
    query_coords = np.stack([query_i, query_j], axis=-1)
    source_coords = np.stack([source_i, source_j], axis=-1)
    tree = spatial.KDTree(source_coords)
    _, ii = tree.query(query_coords, workers=-1)
    nearest_coords = tree.data[ii].astype(int)
    return nearest_coords[:, 0], nearest_coords[:, 1]


def bayer_nearest_neighbor_indices(
    bad_pixel_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nearest valid neighbor indices for bad pixels, respecting the Bayer CFA pattern."""
    valid_mask = ~bad_pixel_mask.astype(bool)
    query_i, query_j = np.where(bad_pixel_mask.astype(bool))
    query_coords = np.stack([query_i, query_j], axis=-1)
    source_i, source_j = np.where(valid_mask)
    source_coords = np.stack([source_i, source_j], axis=-1)

    nearest_i = np.zeros_like(query_i)
    nearest_j = np.zeros_like(query_j)

    for i_offset in [0, 1]:
        for j_offset in [0, 1]:
            source_mask = (source_i % 2 == i_offset) & (source_j % 2 == j_offset)
            channel_source_coords = source_coords[source_mask]

            if channel_source_coords.shape[0] == 0:
                logger.warning("No valid source pixels found for Bayer channel.")
                continue

            tree = spatial.KDTree(channel_source_coords)
            query_mask = (query_i % 2 == i_offset) & (query_j % 2 == j_offset)
            channel_query_coords = query_coords[query_mask]

            if channel_query_coords.shape[0] == 0:
                continue

            _, ii = tree.query(channel_query_coords, workers=-1)
            nearest_coords = tree.data[ii].astype(int)
            nearest_i[query_mask] = nearest_coords[:, 0]
            nearest_j[query_mask] = nearest_coords[:, 1]

    return nearest_i, nearest_j


def interpolate_white_pixels(
    photon_cube: np.ndarray,
    sum_frames: int,
    cfa_mask: np.ndarray,
    hot_pixel_mask: np.ndarray,
) -> np.ndarray:
    """Probabilistic inpainting of masked pixels in a photon cube."""
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    valid_pixel_mask = (~hot_pixel_mask.astype(bool)).astype(np.float32)
    neighbor_counts = ndimage.convolve(valid_pixel_mask, kernel, mode="constant", cval=0)
    masked_photon_cube = photon_cube * valid_pixel_mask[:, :, np.newaxis]
    neighbor_sums = ndimage.convolve(
        masked_photon_cube, kernel[..., np.newaxis], mode="constant", cval=0
    )
    neighbor_avg = np.divide(
        neighbor_sums,
        neighbor_counts[:, :, np.newaxis],
        out=np.zeros_like(neighbor_sums),
        where=neighbor_counts[:, :, np.newaxis] != 0,
    )
    prob_cube = np.clip(neighbor_avg / sum_frames, 0, 1)
    new_values = np.random.binomial(n=sum_frames, p=prob_cube)
    return np.where(cfa_mask[:, :, np.newaxis], new_values, photon_cube)


def colorSPAD_correct_func(
    photon_cube: np.ndarray,
    sum_frames: int,
    colorSPAD_cfa_path: Optional[str],
    hot_pixel_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Geometric corrections and inpainting for the Color SPAD hardware prototype."""
    photon_cube_cropped = np.zeros(
        (254, 496, photon_cube.shape[-1]), dtype=photon_cube.dtype
    )
    photon_cube_cropped[:, :496] = photon_cube[2:, :496]
    photon_cube_cropped[:, 252:256] = photon_cube[2:, 260:264]
    photon_cube_cropped[:, 260:264] = photon_cube[2:, 252:256]
    if colorSPAD_cfa_path:
        cfa = cv2.imread(colorSPAD_cfa_path)[2:, :496, ::-1]
        cfa_mask = cfa.mean(axis=-1) < 255
        return interpolate_white_pixels(
            photon_cube_cropped.astype(int), sum_frames, cfa_mask, hot_pixel_mask
        )
    return photon_cube_cropped


class BaseCube(Dataset):
    """Base dataset for 3-D spatio-temporal data cubes.

    Supports rotations of 0, 90, 180, and 270 degrees plus optional
    horizontal/vertical flips.
    """

    def __init__(
        self,
        chunk_size: int,
        sum_frames: int,
        chunk_stride: Optional[int] = None,
        initial_time_step: int = 0,
        temporal_stride: int = 1,
        rotation: int = 0,
        flip_lr: bool = False,
        flip_ud: bool = False,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.sum_frames = sum_frames
        self.chunk_stride = chunk_stride or self.chunk_size
        self.initial_time_step = initial_time_step
        self.temporal_stride = temporal_stride

        if rotation not in [0, 90, 180, 270]:
            raise ValueError(f"Rotation must be 0, 90, 180, or 270. Got {rotation}")

        self.rotation = rotation
        self.k_rot = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self._h = self._w = self.t_out = self._num_chunks = self.num_time_step = 0

    def _initialize_timing(self, source_total_timesteps: int, num_time_step: int = -1) -> None:
        effective_source_t = source_total_timesteps - self.initial_time_step
        self.num_time_step = (
            effective_source_t
            if num_time_step == -1
            else min(num_time_step, effective_source_t)
        )
        chunk_source_span = (self.chunk_size - 1) * self.temporal_stride + 1
        if self.num_time_step >= chunk_source_span:
            self._num_chunks = (
                floor((self.num_time_step - chunk_source_span) / self.chunk_stride) + 1
            )
        else:
            self._num_chunks = 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._h, self._w, self.t_out

    @property
    def final_time_step(self) -> int:
        return self.initial_time_step + self.num_time_step

    def _determine_dtype(self) -> torch.dtype:
        return torch.uint16 if self.sum_frames > 255 else torch.uint8

    def _apply_transforms(self, cube: Tensor) -> Tensor:
        if self.flip_lr:
            cube = torch.flip(cube, (1,))
        if self.flip_ud:
            cube = torch.flip(cube, (0,))
        if self.k_rot > 0:
            cube = torch.rot90(cube, k=self.k_rot, dims=(0, 1))
        return cube

    def __len__(self) -> int:
        return self._num_chunks


class RealCube(BaseCube):
    """Dataset for real SPAD data from NPY (bit-packed) or H5 files.

    **NPY layout**: ``(T, H, W_packed)`` uint8, each byte encoding 8 binary pixels.

    **H5 layout**: each stored frame is a hardware sum of ``hw_accumulation``
    binary frames (read from file metadata). The loader further groups H5 frames
    so each output frame sums exactly ``sum_frames`` binary frames, where
    ``sum_frames >= hw_accumulation``.

    :param file: Path to ``.npy`` or ``.h5`` data file.
    :param hot_pixel_mask_path: Optional ``.npy`` boolean mask; hot pixels are
        replaced by their nearest Bayer-aware valid neighbour.
    :param colorSPAD_col_correct: Apply colorSPAD hardware column swap correction.
    :param colorSPAD_RGBW_CFA: Path to colorSPAD RGBW CFA image for inpainting.
    :param bayer_pattern: CFA pattern string (``"gray"`` for grayscale sensors).
    """

    def __init__(
        self,
        file: str | Path,
        hot_pixel_mask_path: Optional[str | Path] = None,
        colorSPAD_col_correct: bool = False,
        colorSPAD_RGBW_CFA: Optional[str | Path] = None,
        bayer_pattern: BayerPatternLiteral = "gray",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_path = Path(file)
        self.is_h5 = self.file_path.suffix.lower() == ".h5"
        self.hw_accumulation = 1

        if self.is_h5:
            self._h5_file = h5py.File(self.file_path, "r")
            self._group_path = self._find_data_group_path(self._h5_file)
            if not self._group_path:
                raise ValueError(f"No numeric dataset group found in {file}")

            meta = self.get_ubi_metadata(self.file_path)
            self.hw_accumulation = int(meta.get("sum_frames", 1))
            logger.info(f"H5 hardware accumulation: {self.hw_accumulation} binary frames/stored frame")

            self._group = self._h5_file[self._group_path]
            all_keys = natsorted(self._group.keys())
            self._dataset_keys = all_keys[1:] if len(all_keys) > 1 else all_keys
            if not self._dataset_keys:
                logger.warning("H5 file contains no frames.")

            t = len(self._dataset_keys) * self.hw_accumulation

            if self._dataset_keys:
                first_frame = self._group[self._dataset_keys[0]][...].squeeze()
                self._source_h, self._source_w = first_frame.shape
            else:
                self._source_h, self._source_w = 0, 0
        else:
            self._memmap_cube = np.load(file, mmap_mode="r")
            t, h, w_packed = self._memmap_cube.shape
            self._source_h, self._source_w = h, w_packed * 8

        self.colorSPAD_col_correct = colorSPAD_col_correct
        self.colorSPAD_RGBW_CFA = str(colorSPAD_RGBW_CFA) if colorSPAD_RGBW_CFA else None

        self.bayer_pattern = bayer_pattern.lower()
        if self.bayer_pattern not in ["gray", "grey"]:
            self.bayer_pattern = self._transform_bayer_pattern(
                self.bayer_pattern, self.k_rot, self.flip_lr, self.flip_ud
            )
            logger.info(f"CFA pattern adjusted for rot={self.rotation}: {self.bayer_pattern}")

        self.is_bayer = self.bayer_pattern not in ["gray", "grey"]
        self.t_out = self.chunk_size // self.sum_frames
        self._initialize_timing(t, kwargs.get("num_time_step", -1))
        self.dtype = self._determine_dtype()

        if hot_pixel_mask_path:
            self.hot_pixel_mask = np.load(hot_pixel_mask_path).astype(np.uint8)
            query_i, query_j = np.where(self.hot_pixel_mask.astype(bool))
            if self.is_bayer:
                nearest_i, nearest_j = bayer_nearest_neighbor_indices(self.hot_pixel_mask)
            else:
                nearest_i, nearest_j = nearest_neighbor_indices(self.hot_pixel_mask)
            self.hp_query = (
                torch.from_numpy(query_i).long(),
                torch.from_numpy(query_j).long(),
            )
            self.hp_nearest = (
                torch.from_numpy(nearest_i).long(),
                torch.from_numpy(nearest_j).long(),
            )
        else:
            self.hot_pixel_mask = None

        raw_h, raw_w = (
            (254, 496) if self.colorSPAD_col_correct else (self._source_h, self._source_w)
        )
        self._h, self._w = (raw_w, raw_h) if self.k_rot % 2 != 0 else (raw_h, raw_w)

    def _transform_bayer_pattern(self, p: str, k: int, flr: bool, fud: bool) -> str:
        """Adjust Bayer pattern string for flips and 90-degree CCW rotations.

        Grid indices: 0=TL, 1=TR, 2=BL, 3=BR.
        """
        p_list = list(p.upper())
        if flr:
            p_list[0], p_list[1], p_list[2], p_list[3] = (
                p_list[1], p_list[0], p_list[3], p_list[2]
            )
        if fud:
            p_list[0], p_list[2], p_list[1], p_list[3] = (
                p_list[2], p_list[0], p_list[3], p_list[1]
            )
        old = p_list[:]
        if k == 1:
            p_list = [old[1], old[3], old[0], old[2]]
        elif k == 2:
            p_list = [old[3], old[2], old[1], old[0]]
        elif k == 3:
            p_list = [old[2], old[0], old[3], old[1]]
        return "".join(p_list).lower()

    def _find_data_group_path(self, f) -> Optional[str]:
        found_path = None

        def visitor(name, obj):
            nonlocal found_path
            if found_path:
                return
            if isinstance(obj, h5py.Dataset) and Path(name).name.isdigit():
                found_path = str(Path(name).parent)

        f.visititems(visitor)
        return found_path

    @staticmethod
    def get_ubi_metadata(file_path: str | Path) -> Dict[str, Any]:
        """Read acquisition metadata embedded in H5 attributes."""
        overrides: Dict[str, Any] = {}
        with h5py.File(file_path, "r") as f:
            found_ds = None

            def visitor(name, obj):
                nonlocal found_ds
                if found_ds:
                    return
                if isinstance(obj, h5py.Dataset) and Path(name).name.isdigit():
                    found_ds = name

            f.visititems(visitor)
            if found_ds:
                raw_meta = f[found_ds].attrs.get("metadata")
                if raw_meta:
                    meta = json.loads(
                        raw_meta.decode("utf-8") if isinstance(raw_meta, bytes) else raw_meta
                    )
                    overrides["bayer_pattern"] = meta.get("cfa_pattern", "gray").lower()
                    poly_meta = meta.get("polycam_metadata", {})
                    overrides["sum_frames"] = poly_meta.get(
                        "num_integrated_binary_frames",
                        meta.get("num_integrated_binary_frames", None),
                    )
                    hist = meta.get("node_history", [])
                    if hist and isinstance(hist, list) and len(hist) > 0:
                        sched = hist[0].get("schedule", {}).get("base_schedule", {})
                        if "binary_frame_rate_fps" in sched:
                            overrides["binary_fps"] = sched["binary_frame_rate_fps"]
        return overrides

    def __getitem__(self, index: int) -> Tensor:
        start_idx = self.initial_time_step + index * self.chunk_stride

        if self.is_h5:
            h5_reduce = max(1, self.sum_frames // self.hw_accumulation)
            start_h5 = start_idx // self.hw_accumulation
            num_h5_frames_needed = self.t_out * h5_reduce

            frames = []
            for i in range(num_h5_frames_needed):
                idx = start_h5 + i
                key = self._dataset_keys[min(idx, len(self._dataset_keys) - 1)]
                frames.append(self._group[key][...].squeeze())

            photon_cube_raw = np.stack(frames, axis=-1)
            photon_cube = reduce(
                photon_cube_raw, "h w (t s) -> h w t", "sum", s=h5_reduce
            )
        else:
            end_idx = start_idx + self.chunk_size * self.temporal_stride
            raw_chunk = self._memmap_cube[start_idx:end_idx:self.temporal_stride]
            binary_cube = rearrange(np.unpackbits(raw_chunk, axis=-1), "t h w -> h w t")
            photon_cube = reduce(
                binary_cube, "h w (t s) -> h w t", "sum", s=self.sum_frames
            )

        if self.colorSPAD_col_correct:
            photon_cube = colorSPAD_correct_func(
                photon_cube,
                self.sum_frames,
                self.colorSPAD_RGBW_CFA,
                self.hot_pixel_mask,
            )

        photon_cube = torch.from_numpy(photon_cube.astype(np.int64))

        if self.hot_pixel_mask is not None:
            photon_cube[self.hp_query] = photon_cube[self.hp_nearest]

        return self._apply_transforms(photon_cube).to(self.dtype)



def convert_3_channel_viz_to_raw(bits, bayer_pattern='rggb'):
    if bits.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {bits.shape}")

    h, w, c = bits.shape
    if c != 3:
        raise ValueError(f"Expected at least 3 channels for Bayer conversion, got {c}. (shape: {bits.shape})")

    p = bayer_pattern.lower()
    raw_arr = np.empty((h, w), dtype=bits.dtype)

    if p == "rggb":
        raw_arr[0::2, 0::2] = bits[0::2, 0::2, 0]
        raw_arr[0::2, 1::2] = bits[0::2, 1::2, 1]
        raw_arr[1::2, 0::2] = bits[1::2, 0::2, 1]
        raw_arr[1::2, 1::2] = bits[1::2, 1::2, 2]
    elif p == "bggr":
        raw_arr[0::2, 0::2] = bits[0::2, 0::2, 2]
        raw_arr[0::2, 1::2] = bits[0::2, 1::2, 1]
        raw_arr[1::2, 0::2] = bits[1::2, 0::2, 1]
        raw_arr[1::2, 1::2] = bits[1::2, 1::2, 0]
    elif p == "grbg":
        raw_arr[0::2, 0::2] = bits[0::2, 0::2, 1]
        raw_arr[0::2, 1::2] = bits[0::2, 1::2, 0]
        raw_arr[1::2, 0::2] = bits[1::2, 0::2, 2]
        raw_arr[1::2, 1::2] = bits[1::2, 1::2, 1]
    elif p == "gbrg":
        raw_arr[0::2, 0::2] = bits[0::2, 0::2, 1]
        raw_arr[0::2, 1::2] = bits[0::2, 1::2, 2]
        raw_arr[1::2, 0::2] = bits[1::2, 0::2, 0]
        raw_arr[1::2, 1::2] = bits[1::2, 1::2, 1]
    else:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

    return raw_arr


def load_binary_npy(
    path: str | Path,
    start: int = 0,
    num: int = -1,
    back_to_single_channel=False
) -> list[torch.Tensor]:
    """Load binary frames from a ``.npy`` file via memory-mapping.

    Supports two on-disk layouts:

    1. **Bit-packed** ``(T, H, W_packed)`` uint8 -- each byte holds 8 binary pixels.
    2. **Plain float** ``(H, W, T)`` -- returned as-is per frame.

    :param path: Path to the ``.npy`` file.
    :param start: First frame index to load (0-based).
    :param num: Number of frames to load (``-1`` = all remaining).
    :returns: List of ``(H, W)`` float32 tensors, one per binary frame.
    """
    cube = np.load(str(path), mmap_mode="r")
    cube_ndim = cube.ndim
    if cube_ndim < 3:
        raise ValueError(f"Expected >= 3 dimensional array, got shape {cube.shape}")

    if cube_ndim == 3:
        height, w_packed, total_frames = cube.shape
    elif cube_ndim == 4:
        height, w_packed, channels, total_frames = cube.shape
    else:
        raise ValueError(f"Expected 3-D or 4-D photon cube, got shape {cube.shape}")

    if cube.dtype == np.uint8 and cube.shape[1] < cube.shape[0]: # Check bit-packing
        end = total_frames if num < 0 else min(start + num, total_frames)
        frames: list[torch.Tensor] = []
        for t in tqdm(range(start, end)):
            bits = np.unpackbits(cube[t], axis=1)
            if back_to_single_channel:
                bits = convert_3_channel_viz_to_raw(bits, "rggb")
            frames.append(torch.from_numpy(bits.astype(np.float32)))
        logger.info(f"Loaded {len(frames)} bit-packed binary frames ({height}x{8 * w_packed})")
        return frames
        
    # Not bit-packed, as is:
    end = total_frames if num < 0 else min(start + num, total_frames)
    frames = []
    # print(cube.shape, start, end)
    for t in range(start, end):
        frames.append(torch.from_numpy(cube[:, :, t].astype(np.float32)))
    logger.info(f"Loaded {len(frames)} float frames ({height}x{w_packed})")
    return frames
