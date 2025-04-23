"""Define the processing methods, using (mainly) torchaudio package."""

from typing import Dict

import torch
import torch.nn.functional as F

class Wav2VecProcessing:
    """Wav2vec like processing."""

    def __init__(self, normalize: bool = False):

        self.normalize = normalize

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process the data

        Args:
            wav_form (torch.Tensor): The waveform signal
        """

        feats = waveform.squeeze()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

def create_processing(proc_name: str, **proc_options: Dict):
    """Define feature extraction pipeline.

    Args:
        processing_name (str): The name of the preprocessing class.
        processing_options (dict): The preprocessing options.
    """

    if proc_name == 'Wav2VecProcessing':
        proc_class = Wav2VecProcessing
    else:
        raise ValueError(f"Invalid processing name: {proc_name}")

    proc_spec = proc_class(**proc_options)
    return proc_spec
