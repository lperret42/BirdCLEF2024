import random
import torch
from torch import Tensor
from torchaudio.functional import lowpass_biquad


def add_pink_noise(waveform: Tensor, sr: int, min_amplitude: float = 0.1, max_amplitude: float = 0.4) -> Tensor:
    """
    Adds pink noise to a given waveform. Pink noise is noise with a frequency spectrum that falls off
    at 3 dB per octave (having equal energy per octave).

    Parameters:
    - waveform (Tensor): The input audio signal.
    - sr (int): Sample rate of the audio signal.
    - min_amplitude (float): Minimum amplitude of the pink noise.
    - max_amplitude (float): Maximum amplitude of the pink noise.

    Returns:
    - Tensor: The waveform with added pink noise.
    """
    noise_amplitude = random.uniform(min_amplitude, max_amplitude)
    pink_noise = torch.randn_like(waveform)
    pink_noise = lowpass_biquad(pink_noise, sr, cutoff_freq=1000)
    pink_noise *= noise_amplitude
    return waveform + pink_noise


def add_white_noise(waveform: Tensor, min_amplitude: float = 0.05, max_amplitude: float = 0.12) -> Tensor:
    """
    Adds white noise to a given waveform. White noise has a constant power density across all frequencies.

    Parameters:
    - waveform (Tensor): The input audio signal.
    - min_amplitude (float): Minimum amplitude of the white noise.
    - max_amplitude (float): Maximum amplitude of the white noise.

    Returns:
    - Tensor: The waveform with added white noise.
    """
    noise_amplitude = random.uniform(min_amplitude, max_amplitude)
    white_noise = torch.randn_like(waveform)
    white_noise *= noise_amplitude
    return waveform + white_noise


def apply_augmentations(waveform: Tensor, sr: int) -> Tensor:
    """
    Applies random audio augmentations (pink noise or white noise) to the input waveform.

    Parameters:
    - waveform (Tensor): The input audio signal.
    - sr (int): Sample rate of the audio signal.

    Returns:
    - Tensor: The augmented waveform.
    """
    if random.random() <= 0.5:
        waveform = add_pink_noise(waveform, sr)
    if random.random() <= 0.5:
        waveform = add_white_noise(waveform)
    return waveform
