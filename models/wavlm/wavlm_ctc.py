import sys
from pathlib import Path

from typing import Tuple, Optional, Dict

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from wavlm.WavLM import WavLM, WavLMConfig


class WavLMCTC(nn.Module):
    """Define the WavLmCTC model."""

    def __init__(self, model: nn.Module, cfg: dict):
        super(WavLMCTC, self).__init__()

        self.cfg = cfg
        self.wavlm = model
        self.num_labels = cfg.num_labels

        if cfg.final_dropout != 0.:
            self.final_dropout = nn.Dropout(cfg.final_dropout)
        else:
            self.final_dropout = None

        self.mask = ( self.wavlm.mask_prob>0. or self.wavlm.mask_channel_prob>0. )
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.classifier = nn.Linear(self.encoder_embed_dim, self.num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

        # freeze the feature extractor
        for name, param in self.wavlm.named_parameters():
            if 'feature_extractor' in name:
                param.requires_grad = False

    def freeze_parameters(self, base: bool=True) -> None:
        """Freeze the base w2v parameters.

        Args:
            base (bool, optional): Freez the encoder or not.
        """

        for name, param in self.wavlm.named_parameters():
            if not 'feature_extractor' in name:
                if base:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def process(self,
                waveforms: torch.Tensor,
                waveform_sizes: torch.Tensor,
                processing) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the waveforms

        Args:
            waveforms (torch.Tensor): The (possibly augmented) waveforms
            waveform_sizes (torch.Tensor): waveform sizes
            processing: Processing to be applied to the waveforms.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed waveforms
        """

        features = []
        padding_masks = []
        for i, wf in enumerate(waveforms):
            feats = processing(wf[:waveform_sizes[i]])
            features.append(feats)
            padding_masks.append(torch.BoolTensor(feats.size(0)).fill_(False))

        features = pad_sequence(features, batch_first=True, padding_value=0.0)
        padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=True)

        return features, padding_masks


    def forward(self, source: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward the data.

        Args:
            source (torch.Tensor): The processed waveforms.
            padding_mask (torch.Tensor): The padding mask.

        Returns:
            torch.Tensor: token logits and the padding mask.
        """

        mask = self.mask and self.training
        x, padding_mask = self.wavlm.extract_features(source=source, mask=mask, padding_mask=padding_mask)

        if self.final_dropout:
            x = self.final_dropout(x)
        logits = self.classifier(x)
        return logits, padding_mask


    def process_batch(self,
                      waveforms: torch.Tensor,
                      waveform_sizes:torch.Tensor,
                      processing,
                      device: torch.device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the batch

        Args:
            waveforms (torch.Tensor): Tensor of waveforms
            waveform_sizes (torch.Tensor): Tensor of waveform sizes
            processing (_type_): processing type
            device (torch.device, optional): device type. Defaults to 'cpu'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """

        inputs, padding_masks = self.process(waveforms, waveform_sizes, processing)
        
        inputs = inputs.to(device)
        padding_masks = padding_masks.to(device)
        logits, padding_masks = self.forward(inputs, padding_masks)
        input_sizes = (padding_masks == False).sum(dim=1)
        input_sizes = input_sizes.to(device)
        return logits, input_sizes

if __name__ == '__main__':

    from omegaconf import OmegaConf

    PRETRAINED_PATH = r'path/to/the/pretrained/WavLM-Base.pt'
    CONFIG_PATH = r'path/to/the/yaml/config/wavlm_ctc.yaml'

    config = OmegaConf.load(CONFIG_PATH)
    model_config = config.model
    arg_overrides = model_config.arg_overrides
    

    checkpoint = torch.load(PRETRAINED_PATH)
    model_cfg = checkpoint['cfg']
    model_cfg.update(arg_overrides)
    print(model_cfg)
    model_cfg = WavLMConfig(model_cfg)
    wavlm = WavLM(model_cfg)
    wavlm.load_state_dict(checkpoint['model'])
    model_cfg.update(model_config.arg_ctc)
    model = WavLMCTC(wavlm, model_cfg)
    print(model)
    
