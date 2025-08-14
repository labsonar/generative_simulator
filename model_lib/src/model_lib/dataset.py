"""
Module to load audio detections and background data as PyTorch datasets.

This module defines tools for segmenting WAV files into blocks and loading them
as a PyTorch-compatible dataset. It includes an abstract interface for audio
block processing, a concrete window-based segmentation class, and a dataset class
that reads audio files from a directory and maps them into labeled examples.

To prepare audio files for this dataset, run: `test/prepare_dataset.py`.
"""

import typing
import abc
import pathlib
import wave

import numpy as np
import pandas as pd

import torch
import torch.utils.data as torch_data

import lps_utils.utils as lps_utils


class DataProcessing(abc.ABC):
    """
    Abstract base class for processing audio data into blocks.
    """

    @abc.abstractmethod
    def get_n_blocks(self, wav_file: wave.Wave_read) -> int:
        """
        Return the number of blocks that can be extracted from the WAV file.

        Args:
            wav_file (wave.Wave_read): Opened WAV file handle.

        Returns:
            int: Number of extractable blocks.
        """

    @abc.abstractmethod
    def get(self, wav_file: wave.Wave_read, block_id: int) -> np.ndarray:
        """
        Extract a specific block of audio data from the WAV file.

        Args:
            wav_file (wave.Wave_read): Opened WAV file handle.
            block_id (int): Block index to extract.

        Returns:
            np.ndarray: Audio block as a NumPy array.
        """


class SplitWindow(DataProcessing):
    """
    Splits a WAV file into overlapping or non-overlapping fixed-size windows.
    """

    def __init__(self, window: int, overlap: int):
        """
        Initialize SplitWindow processor.

        Args:
            window (int): Size of each window in samples.
            overlap (int): Number of overlapping samples between windows.
        """
        self.window = window
        self.overlap = overlap
        self.hop_size = window - overlap

    def get_n_blocks(self, wav_file: wave.Wave_read) -> int:
        """
        Return number of blocks available in the WAV file using window and overlap.

        Args:
            wav_file (wave.Wave_read): Opened WAV file.

        Returns:
            int: Number of extractable windows.
        """
        total_frames = wav_file.getnframes()
        if total_frames < self.window:
            return 0
        return int((total_frames - self.window) // self.hop_size + 1)

    def get(self, wav_file: wave.Wave_read, block_id: int) -> torch.Tensor:
        """
        Extract the `block_id`-th window from the WAV file.

        Args:
            wav_file (wave.Wave_read): Opened WAV file.
            block_id (int): Index of the block to extract.

        Returns:
            np.ndarray: Extracted audio data as a NumPy array.
        """
        start = block_id * self.hop_size
        wav_file.setpos(start)
        raw_data = wav_file.readframes(self.window)

        sample_width = wav_file.getsampwidth()
        n_channels = wav_file.getnchannels()

        dtype = {
            1: np.uint8,   # WAV 8-bit: unsigned
            2: np.int16,   # WAV 16-bit: signed
            3: np.int32,   # WAV 24-bit: handled as 32-bit
            4: np.int32    # WAV 32-bit: signed
        }.get(sample_width)

        if dtype is None:
            raise ValueError(f"Sample width {sample_width} bytes not supported.")

        raw_data = np.frombuffer(raw_data, dtype=dtype)

        if n_channels > 1:
            raw_data = raw_data.reshape(-1, n_channels)

        tensor = torch.from_numpy(raw_data.copy()).float()

        if sample_width == 1: #unsigned
            # uint8: convert from [0, 255] to [-1, 1]
            tensor = (tensor - 128.0) / 128.0
        else: #signed
            max_val = float(2 ** (8 * sample_width - 1))
            tensor = tensor / max_val

        return tensor


class FolderDataset(torch_data.Dataset):
    """
    A PyTorch Dataset for loading and processing blocks from WAV files in a folder.

    Each file is assumed to belong to a target class (derived from its folder name),
    and split into fixed-size blocks using a DataProcessing instance.
    """

    @staticmethod
    def _default_extract_target_id(path: pathlib.PurePath) -> str:
        """
        Default method to extract target class from file path.

        Args:
            path (pathlib.PurePath): Path to the audio file.

        Returns:
            str: Class name (folder name).
        """
        return path.parent.name

    @staticmethod
    def _default_extract_file_id(path: pathlib.PurePath) -> str:
        """
        Default method to extract file ID from file path.

        Args:
            path (pathlib.PurePath): Path to the audio file.

        Returns:
            str: File ID (filename without extension).
        """
        return path.stem

    def __init__(self,
                 base_dir: str,
                 processing: DataProcessing,
                 extract_target_id: typing.Callable[[pathlib.PurePath], str] =
                        _default_extract_target_id,
                 extract_file_id: typing.Callable[[pathlib.PurePath], str] =
                        _default_extract_file_id):
        """
        Initialize the dataset from a base directory of audio files.

        Args:
            base_dir (str): Directory containing WAV files.
            processing (DataProcessing): Processor to extract data blocks.
            extract_target_id (Callable): Function to extract class label from path.
            extract_file_id (Callable): Function to extract file ID from path.
        """
        self.class_list = []
        self.files = []
        self.processing = processing
        self.limit_ids = [0]

        files = lps_utils.find_files(base_dir)
        for filename in files:
            path = pathlib.Path(filename)
            file_id = extract_file_id(path)
            target_id = extract_target_id(path)

            if target_id not in self.class_list:
                self.class_list.append(target_id)

            wav_file = wave.open(filename)
            n_blocks = processing.get_n_blocks(wav_file)
            wav_file.close()

            if n_blocks > 0:
                self.limit_ids.append(self.limit_ids[-1] + n_blocks)

                self.files.append({
                    "id": file_id,
                    "filename": filename,
                    "target": target_id,
                    "n_blocks": n_blocks
                })

        self.class_list.sort()

        self.df = pd.DataFrame(self.files)[["id", "n_blocks", "filename", "target"]]

    def __str__(self) -> str:
        return str(self.df)

    def __len__(self):
        return self.limit_ids[-1]

    def __getitem__(self, index):
        current_index = next(i for i, limit in enumerate(self.limit_ids) if limit > index) - 1
        offset_index = index - self.limit_ids[current_index]

        file = self.files[current_index]

        wav_file = wave.open(file["filename"])
        raw_data = self.processing.get(wav_file, offset_index)
        wav_file.close()

        target_index = self.class_list.index(file["target"])

        return raw_data, target_index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process data from a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing the data")
    args = parser.parse_args()

    dataset = FolderDataset(base_dir=args.folder,
                            processing=SplitWindow(22050, 0))

    print(len(dataset))
    print(dataset.class_list)
    print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        32,
        shuffle=True)

    for sample_data, _ in dataloader:
        print(sample_data.shape)
        break
