import os
from typing import Optional, Tuple, List, Union
import random
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .wave_aug import apply_augmentations


class BirdDataset(Dataset):
    """
    A dataset class for bird audio processing for different modes including training, validation, and inference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        num_classes: Optional[int] = None,
        sr: int = 32000,
        duration: Union[int, List[float]] = 5,
        split: Optional[str] = None,
        mode: str = 'valid',
        mixup_prob: Optional[float] = None,
        mixup_max_num: int = 2,
        cache_audio_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.df = df
        self.audio_dir = audio_dir
        self.num_classes = num_classes
        self.sr = sr
        self.duration = duration
        self.split = split
        self.mode = mode
        self.mixup_prob = mixup_prob
        self.mixup_max_num = mixup_max_num
        self.cache_audio_dir = cache_audio_dir
        self.verbose = verbose
        self.one_sample_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def _getitem(self, idx: int, duration: int) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[float]]:
        """
        Retrieves an item from the dataset at the specified index.
        Handles audio loading, resampling, caching, and optional mixup.
        """
        row = self.df.iloc[idx]
        _id = row['row_id']
        filename = row['filename']
        audio_path = os.path.join(self.audio_dir, filename)
        label_ohe = self._get_label(row) if self.mode != 'inference' else None

        waveform = self._load_waveform(audio_path, filename, duration)

        waveform = self._handle_duration(waveform, duration, row)

        if self.mode == 'train':
            waveform = apply_augmentations(waveform, self.sr)

        weight = row.get('weight', None)
        return _id, waveform, label_ohe, weight

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[float]]:
        """
        Public method to get a dataset item, handling variable durations and mixup logic.
        """
        duration = self._resolve_duration()
        _id, waveform, label_ohe, weight = self._getitem(idx, duration)

        if self.mixup_prob and random.random() <= self.mixup_prob:
            _id, waveform, label_ohe, weight = self._apply_mixup(_id, waveform, label_ohe, weight, duration)

        return _id, waveform, label_ohe, weight

    def _resolve_duration(self) -> int:
        """
        Resolve the duration for the audio sample, either a fixed value or randomly between two.
        """
        if isinstance(self.duration, (int, float)):
            return int(self.duration)
        elif isinstance(self.duration, list) and len(self.duration) == 2:
            return int(random.uniform(self.duration[0], self.duration[1]))
        else:
            raise ValueError("Duration must be an int, a float, or a list of two floats.")

    def _get_label(self, row) -> torch.Tensor:
        """
        Get the one-hot encoded label for training and validation modes.
        """
        label = row['target']
        label_ohe = torch.zeros(self.num_classes, dtype=torch.float32)
        label_ohe[label] = 1.0
        return label_ohe

    def _load_waveform(self, audio_path: str, filename: str, duration: int) -> torch.Tensor:
        """
        Loads waveform from cache or file system, applying necessary transformations.
        """
        audio_as_tensor_path = self._get_cached_path(filename)
        waveform = self._handle_audio_loading(audio_path, audio_as_tensor_path)
        return waveform

    def _handle_audio_loading(self, audio_path: str, cached_path: Optional[str]) -> torch.Tensor:
        """
        Load audio from a file and cache it, or load from cache if already available.
        """
        if cached_path and os.path.exists(cached_path):
            waveform = torch.load(cached_path)
            waveform = waveform.to(torch.float32)
        else:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sr:
                waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
            if cached_path:
                torch.save(waveform.to(torch.float16), cached_path)
        return waveform

    def _handle_duration(self, waveform: torch.Tensor, duration: int, row: pd.Series) -> torch.Tensor:
        """
        Adjust waveform duration
        """
        # Inference mode processing for exact matching duration segments
        if self.mode == 'inference':
            start_sec = row['start']
            end_sec = row['end']
            start_sample = int(start_sec * self.sr)
            end_sample = int(end_sec * self.sr)
            waveform = waveform[:, start_sample:end_sample]
        else:
            # Handle random or first split logic for training/validation
            total_samples = waveform.size(1)
            max_start = total_samples - self.sr * duration
            if max_start > 0:
                if self.split == 'random':
                    start_sample = random.randint(0, max_start)
                elif self.split == 'first':
                    start_sample = 0
                waveform = waveform[:, start_sample:start_sample + self.sr * duration]
            else:
                # If the waveform is shorter than the required duration, pad it
                padding_needed = self.sr * duration - total_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform

    def _apply_mixup(self, _id: str, waveform: torch.Tensor, label_ohe: torch.Tensor, weight: Optional[float], duration: int) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[float]]:
        """
        Apply mixup augmentation to the waveform and labels.
        """
        n = random.randint(1, self.mixup_max_num)
        mixup_coef = 1 / (n + 1)
        mixed_waveform, mixed_label_ohe, mixed_weight = waveform * mixup_coef, label_ohe * mixup_coef, weight * mixup_coef if weight is not None else None

        for _ in range(n):
            idx_mix = random.randint(0, self.__len__() - 1)
            _, waveform_mix, label_ohe_mix, weight_mix = self._getitem(idx_mix, duration)
            mixed_waveform += mixup_coef * waveform_mix
            mixed_label_ohe += mixup_coef * label_ohe_mix
            mixed_weight = mixed_weight + mixup_coef * weight_mix if weight is not None else None

        return _id, mixed_waveform, mixed_label_ohe, mixed_weight

    def _get_cached_path(self, filename: str) -> Optional[str]:
        """
        Construct the path for cached audio tensor based on filename.
        """
        if self.cache_audio_dir:
            return os.path.join(self.cache_audio_dir, filename.replace('/', '_').replace('.ogg', '.pt').replace('.wav', '.pt'))
        return None


def collate_fn(batch: List[Tuple[str, torch.Tensor, torch.Tensor, Optional[float]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Custom collate function to handle batching of items.
    """
    _ids, waveforms, label_ohes, weights = zip(*batch)
    _ids = default_collate(_ids)
    waveforms = default_collate(waveforms)
    label_ohes = default_collate(label_ohes) if label_ohes[0] is not None else None
    weights = default_collate(weights) if weights[0] is not None else None
    return _ids, waveforms, label_ohes, weights
