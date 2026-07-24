import bisect
import json
import logging
import math
import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def _read_h5_rows(dataset, rows):
    if len(rows) == 1 or all(
        current == previous + 1
        for previous, current in zip(rows, rows[1:])
    ):
        return np.asarray(dataset[rows[0] : rows[-1] + 1, ...])

    output = np.empty((len(rows),) + dataset.shape[1:], dtype=dataset.dtype)
    for output_index, row in enumerate(rows):
        output[output_index] = dataset[row, ...]
    return output


class H5PanoramicVideoDataset(Dataset):
    def __init__(
        self,
        json_files,
        json_root,
        h5_root,
        resolution=(1024, 512),
        clip_len=25,
        original_fps=20,
        target_fps_range=(1, 20),
    ):
        """Load panoramic video clips from preprocessed HDF5 files.

        :param json_files: JSON filenames to load, for example
            ``["setting_dynamic.json"]``.
        :param json_root: Root directory containing the JSON files.
        :param h5_root: Root directory containing the preprocessed HDF5 files.
        :param resolution: Output resolution as ``(width, height)``.
        :param clip_len: Number of frames returned for each clip.
        :param original_fps: Original frame rate of the data.
        :param target_fps_range: Inclusive range used for random FPS sampling.
        """
        self.h5_root = h5_root
        self.resolution = resolution
        self.clip_len = clip_len
        self.original_fps = original_fps
        self.target_fps_range = target_fps_range

        # Store each clip as an (h5_file_path, clip_length) tuple.
        self.clips = []
        self.clip_lengths = []
        self.cumulative_lengths = [0]

        total_frames = 0
        excluded_towns = {"town02", "town10", "town15"}

        print("Initializing H5 Dataset...")
        for json_file_name in json_files:
            json_path = os.path.join(json_root, json_file_name)

            with open(json_path, "r") as file:
                data = json.load(file)

            for town, paths in data.items():
                if town in excluded_towns:
                    logger.info("Skipping town: %s", town)
                    continue

                for path, clips_in_path in paths.items():
                    for clip_name, frames in clips_in_path.items():
                        clip_actual_len = len(frames)
                        if clip_actual_len < self.clip_len:
                            continue

                        h5_filename = f"{town}_{path}_{clip_name}.h5"
                        h5_file_path = os.path.join(self.h5_root, h5_filename)

                        if not os.path.exists(h5_file_path):
                            raise FileNotFoundError(
                                "Required H5 file is missing: "
                                f"{h5_file_path} "
                                f"(town={town}, path={path}, clip={clip_name})"
                            )

                        self.clips.append((h5_file_path, clip_actual_len))
                        self.clip_lengths.append(clip_actual_len)
                        total_frames += clip_actual_len
                        self.cumulative_lengths.append(total_frames)

        self.total_frames = total_frames
        if not self.clips:
            print(
                "ERROR: No valid H5 clips were found. "
                "Check h5_root and json_root paths."
            )
        print(
            f"Dataset initialized: {len(self.clips)} clips loaded, "
            f"{self.total_frames} total frames."
        )

    def __len__(self):
        # Weight clip sampling in proportion to the number of frames.
        return self.total_frames

    def find_clip_index(self, idx):
        return bisect.bisect_right(self.cumulative_lengths, idx) - 1

    def __getitem__(self, idx):
        clip_index = self.find_clip_index(idx)
        h5_path, clip_actual_len = self.clips[clip_index]

        target_fps = random.randint(
            self.target_fps_range[0], self.target_fps_range[1]
        )
        current_frame_skip = max(1, round(self.original_fps / target_fps))

        required_len = (self.clip_len - 1) * current_frame_skip + 1
        if required_len > clip_actual_len:
            max_allowed_skip = math.floor(
                (clip_actual_len - 1) / (self.clip_len - 1)
            )
            current_frame_skip = max(1, max_allowed_skip)
            target_fps = round(self.original_fps / current_frame_skip)
            logger.debug(
                "Adjusted skip to %s (target_fps ~%s) for clip %s (len=%s)",
                current_frame_skip,
                target_fps,
                h5_path,
                clip_actual_len,
            )

        max_start_idx = (
            clip_actual_len - (self.clip_len - 1) * current_frame_skip - 1
        )
        assert max_start_idx >= 0, "Error: max_start_idx calculation failed."

        start_index = random.randint(0, max_start_idx)
        actual_indices_in_clip = [
            start_index + i * current_frame_skip for i in range(self.clip_len)
        ]

        with h5py.File(h5_path, "r") as file:
            rgb_data = _read_h5_rows(file["rgb"], actual_indices_in_clip)
            depth_data = _read_h5_rows(file["depth"], actual_indices_in_clip)

        depth_data = np.minimum(depth_data, 1000.0)

        video_tensor = torch.from_numpy(rgb_data).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = F.resize(
            video_tensor,
            size=(self.resolution[1], self.resolution[0]),
            antialias=True,
        )

        raw_depth_tensor = torch.from_numpy(depth_data).float()
        raw_depth_tensor = raw_depth_tensor.permute(0, 3, 1, 2)
        raw_depth_tensor = F.resize(
            raw_depth_tensor,
            size=(self.resolution[1], self.resolution[0]),
            interpolation=F.InterpolationMode.NEAREST,
        )

        disparity_tensor = 1.0 / (raw_depth_tensor + 1e-6)
        min_disparity = torch.min(disparity_tensor)
        max_disparity = torch.max(disparity_tensor)

        if max_disparity > min_disparity:
            disparity_normalized = (disparity_tensor - min_disparity) / (
                max_disparity - min_disparity + 1e-6
            )
        else:
            disparity_normalized = torch.zeros_like(disparity_tensor)

        depth_tensor_processed = disparity_normalized.repeat(1, 3, 1, 1)

        return {
            "video": video_tensor,
            "depth": depth_tensor_processed,
            "target_fps": float(target_fps),
        }
