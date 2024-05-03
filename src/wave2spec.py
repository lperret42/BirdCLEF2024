import torch
import torchaudio.transforms as T


class MelSpectrogram(torch.nn.Module):
    """
    A wrapper class for torchaudio's MelSpectrogram.

    This class facilitates the generation of mel spectrograms from audio waveforms by encapsulating the torchaudio's MelSpectrogram.
    It automatically handles device assignments for computations either on CPU or GPU.

    Attributes:
        sample_rate (int): The sample rate of the audio (in Hz).
        n_fft (int): Number of FFT components.
        win_length (int): Each frame of audio is windowed by `win_length` samples.
        hop_length (int): Number of samples between successive frames.
        f_min (float): The lowest frequency (in Hz).
        f_max (float): The highest frequency (in Hz).
        n_mels (int): Number of mel bands to generate.
        power (float): Exponent for the magnitude melspectrogram.
        device (str): Device type ('cpu' or 'cuda').
    """

    def __init__(self, sample_rate: int = 32000, n_fft: int = 2048, win_length: int = 640, hop_length: int = 640,
                 f_min: float = 0, f_max: float = 16000, n_mels: int = 256, power: float = 2.0, device: str = 'cpu') -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.power = power
        self.device = device

        self.spec_obj = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
        )

        self.to(self.device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Processes the waveform input into mel spectrogram with log scale.

        Args:
            waveform (torch.Tensor): Tensor representing the audio waveform.

        Returns:
            torch.Tensor: Mel spectrogram in dB scale.
        """
        waveform = waveform.to(self.device)
        waveform = waveform.to(torch.float32) if str(self.device) == 'cpu' else waveform

        mel_spec = self.spec_obj(waveform)
        mel_spec_db = 20 * torch.log10(torch.clamp(mel_spec, min=1e-5))

        return mel_spec_db
