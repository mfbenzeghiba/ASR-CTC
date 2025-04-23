"""Define the wav2vec2_ctc model."""

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class Wav2Vec2CTC(nn.Module):
    """Define Wav2vec2 with CTC."""

    def __init__(self, model: nn.Module, cfg: dict):
        super(Wav2Vec2CTC, self).__init__()

        self.cfg = cfg
        self.w2v = model
        self.num_labels = cfg.num_labels
        self.final_dropout = cfg.get('final_dropout', None)
        if self.final_dropout is not None and self.final_dropout != 0.:
            self.final_dropout = nn.Dropout(self.final_dropout)

        self.mask = ( self.w2v.mask_prob>0. or self.w2v.mask_channel_prob>0. )
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.classifier = nn.Linear(self.encoder_embed_dim, self.num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        # freeze the feature extractor
        for name, param in self.w2v.named_parameters():
            if name == 'feature_extractor':
                param.requires_grad = False

    def freeze_parameters(self, base: bool=True) -> None:
        """Freeze the base w2v parameters.

        Args:
            base (bool, optional): Freez the encoder or not.
        """

        for name, param in self.w2v.named_parameters():
            if name != 'feature_extractor':
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

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute the mask after the feature extraction.

        Args:
            features (torch.Tensor): The extracted features from the cnn
            padding_mask (torch.Tensor): The original padding mask on the waveforms.

        Returns:
            torch.Tensor: The padding mask
        """

        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(self, source: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward the data.

        Args:
            x (torch.Tensor): features
            padding_mask (torch.Tensor): The padding mask for the features

        Returns:
            torch.Tensor: _description_
        """

        with torch.no_grad():
            features = self.w2v.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.w2v.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.w2v.post_extract_proj is not None:
            features = self.w2v.post_extract_proj(features)

        if self.mask and self.training:
            x, _ = self.w2v.apply_mask(features, padding_mask)
        else:
            x = features
        x, _ = self.w2v.encoder(x, padding_mask=padding_mask)
        if self.final_dropout:
            x = self.final_dropout(x)
        logits = self.classifier(x)
        return logits, padding_mask


    def process_batch(self,
                      waveforms: torch.Tensor,
                      waveform_sizes:torch.Tensor,
                      processing,
                      device: torch.device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch

        Args:
            waveforms (torch.Tensor): Tensor of waveforms
            waveform_sizes (torch.Tensor): Tensor of waveform sizes
            processing (_type_): processing type
            device (torch.device, optional): device. default to 'cpu'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
        """

        # process the data
        inputs, padding_masks = self.process(waveforms, waveform_sizes, processing)
        inputs = inputs.to(device)
        padding_masks = padding_masks.to(device)
        logits, padding_masks = self.forward(inputs, padding_masks)
        input_sizes = (padding_masks == False).sum(dim=1)
        input_sizes = input_sizes.to(device)
        return logits, input_sizes


if __name__ == '__main__':

    from omegaconf import OmegaConf, open_dict
    import fairseq

    PRETRAINED_MODEL = 'path/to/pretrained/wav2vec_small.pt'
    CONFIG_PATH = 'path/to/config/wav2vec_ctc.yaml'

    config = OmegaConf.load( CONFIG_PATH )
    mcfg = config.model
    print(mcfg)
    arg_overrides = mcfg.arg_overrides
    model, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([PRETRAINED_MODEL],
                                                                          arg_overrides)
    wav2vec2 = model[0]
    model_cfg = cfg.model
    wav2vec2.remove_pretraining_modules(last_layer=mcfg.encoder_layers-1)

    # updata model config with CTC parameters
    with open_dict(model_cfg):
        for k in mcfg.arg_ctc:
            model_cfg[k] = mcfg.arg_ctc[k]
        model = Wav2Vec2CTC(wav2vec2, model_cfg)
    print(model)
