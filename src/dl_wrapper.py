from typing import Iterator, Tuple, Any
import torch


class DataLoaderWrapper(object):
    """
    A wrapper for a PyTorch DataLoader that enables GPU-based transformations on the data.

    This class is designed to handle situations where PyTorch DataLoader threads may not efficiently handle
    CUDA computations directly. It applies transformations to the data using a provided wave-to-spectrogram
    conversion object, typically expected to execute on the GPU.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The underlying dataloader that provides batches of data.
        wave2spec_obj (Callable): A callable that converts waveform data to spectrogram data, likely utilizing CUDA.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, wave2spec_obj: Any) -> None:
        """
        Initializes the DataLoaderWrapper with a specified dataloader and a transformation object.

        Parameters:
            dataloader (torch.utils.data.DataLoader): The dataloader to wrap.
            wave2spec_obj (Callable): The transformation object that converts waveforms to spectrograms.
        """
        self.dataloader = dataloader
        self.wave2spec_obj = wave2spec_obj

    def __iter__(self) -> Iterator[Tuple[Any, Any, Any, Any]]:
        """
        Allows the DataLoaderWrapper to be iterable, yielding transformed data batches.

        Yields:
            Tuple[Any, Any, Any, Any]: A tuple containing the batch ID, transformed spectrogram, labels, and any
                                       additional data, which are directly passed through from the DataLoader.
        """
        for _id, wave, y, r in self.dataloader:
            spec = self.wave2spec_obj(wave)
            yield _id, spec, y, r

    def __len__(self) -> int:
        """
        Returns the number of batches available in the DataLoader.

        Returns:
            int: The total number of batches in the underlying DataLoader.
        """
        return len(self.dataloader)

