from typing import Tuple, Dict

from omegaconf import OmegaConf, open_dict

import torch
import fairseq
from models.wav2vec.wav2vec2_ctc import Wav2Vec2CTC
from models.hubert.hubert_ctc import HubertCTC
from models.wavlm.WavLM import WavLM, WavLMConfig
from models.wavlm.wavlm_ctc import WavLMCTC

def load_wav2vec_model(pretrained_path: str,
                       arg_overrides: OmegaConf) -> Tuple[torch.nn.Module, OmegaConf]:
    """Load the pre-trained model.

    Args:
        pretrained_path (str): The path to the pretrained model.
        arg_overrides (OmegaConf): arguments in the pretrained model to be overrided

    Returns:
        pretrained model and the model config
    """

    model, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_path],
                                                                          arg_overrides)
    wav2vec2 = model[0]
    model_cfg = cfg.model
    return wav2vec2, model_cfg


def load_wavlm_model(pretrained_path: str,
                       arg_overrides: OmegaConf) -> Tuple[torch.nn.Module, OmegaConf]:
    """Load the pre-trained model.

    Args:
        pretrained_path (str): The path to the pretrained model.
        arg_overrides (OmegaConf): arguments in the pretrained model to be overrided

    Returns:
        pretrained model and the model config
    """

    checkpoint = torch.load(pretrained_path)
    model_cfg = WavLMConfig(checkpoint['cfg'])
    model_cfg.update(arg_overrides)
    wavlm = WavLM(model_cfg)
    wavlm.load_state_dict(checkpoint['model'], strict=False)
    return wavlm, model_cfg
    

def create_model(cfg: Dict) -> torch.nn.Module:
    """Implement the model.

    Args:cfg (Dict): Dictionary contains the model options.

    Returns:
        torch.nn.Module: the model
    """

    from_pretrained = cfg.from_pretrained
    model_type = cfg.type
    model = None

    if model_type == 'wav2vec':
        wv2, model_cfg = load_wav2vec_model(from_pretrained, cfg.arg_overrides)
        wv2.remove_pretraining_modules()
        with open_dict(model_cfg):
            for k in cfg.arg_ctc:
                model_cfg[k] = cfg.arg_ctc[k]
        model = Wav2Vec2CTC(wv2, model_cfg)
    if model_type == 'hubert':
        hubert, model_cfg = load_wav2vec_model(from_pretrained, cfg.arg_overrides)
        hubert.remove_pretraining_modules()
        with open_dict(model_cfg):
            for k in cfg.arg_ctc:
                model_cfg[k] = cfg.arg_ctc[k]
        model = HubertCTC(hubert, model_cfg)
    if model_type == 'wavlm':
        wavlm, model_cfg = load_wavlm_model(from_pretrained, cfg.arg_overrides)
        model_cfg.update(cfg.arg_ctc)
        model = WavLMCTC(wavlm, model_cfg)

    return model


def load_checkpoint(model_path: str) -> Tuple[torch.nn.Module, Dict]:
    """Load a previously trained model.

    Args:
        model_path (str, optional): The path to the model.
    """

    checkpoint = torch.load(model_path, map_location='cpu')
    model_cfg = checkpoint['cfg']
    train_resume = {}
    train_resume['epoch'] = checkpoint.get('epoch', None)
    train_resume['epoch_loss'] = checkpoint.get('epoch loss', None)
    train_resume['valid_error'] = checkpoint.get('valid_error', None)
    train_resume['learning_rate'] = checkpoint.get('learning_rate', None)
    train_resume['optimizer'] = checkpoint.get('optimizer', None)
    train_resume['scheduler'] = checkpoint.get('scheduler', None)

    model = create_model(model_cfg)
    model.load_state_dict(checkpoint['state_dict'])
    return model, train_resume
