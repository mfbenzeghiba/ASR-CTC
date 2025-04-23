"""Script to evaluate the model."""

import os
from pathlib import Path
import sys
import argparse
import logging
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from data_providers.timit.timit import TimitTokenizer, TimitDataset
from utils.models_utils import load_checkpoint
from utils.metrics import greedy_decoding, compute_cer
from data_transforms import processing

def compute_perfs(batch, model, preprocess, idx2token, blank_idx, device):

    _, waveforms, waveform_sizes, targets, target_sizes = batch

    with torch.no_grad():
        logits, input_size = model.process_batch(
                waveforms=waveforms,
                waveform_sizes=waveform_sizes,
                processing=preprocess,
                device=device
        )
        log_probs = F.log_softmax(logits, dim=2).detach()

        _, char_hyps = greedy_decoding(log_probs.cpu(), input_size, idx2token, blank_idx)
        char_refs = []
        for i, _ in enumerate(targets):
            seq_labels = targets[i][:target_sizes[i]].tolist()
            char_refs.append([idx2token[k] for _, k in enumerate(seq_labels)])
        
        char_errors = compute_cer(char_refs, char_hyps)
        
        word_hyps = [''.join(h).split() for i, h in enumerate(char_hyps) ]
        word_refs = [''.join(r).split() for i, r in enumerate(char_refs) ]
        word_errors = compute_cer(word_refs, word_hyps)
        return char_errors, word_errors

def main(opts, eval_data, checkpoint):

    torch.manual_seed(7)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.info("-" * 50)
    logging.info('PyTorch Version: %s', torch.__version__)
    logging.info("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_options = opts.datasets
    tokenizer = TimitTokenizer(data_options)

    blank_idx = tokenizer.blank_idx
    idx2token = tokenizer.index_to_token

    exp_folder = opts.exp_dir

    model_options = opts.model
    model_folder = model_options.checkpoint_dir

    model_path = os.path.join(exp_folder, model_folder, checkpoint)
    print(model_path)
    model, _ = load_checkpoint(model_path)
    model.to(device)
    model.eval()
    
    proc_config = model_options.processing
    proc_name = proc_config.name
    proc_options = proc_config.options
    preprocess = processing.create_processing(proc_name, **proc_options)

    #eval_dataset = LibriLightDataset(eval_data, tokenizer)
    eval_dataset = TimitDataset(eval_data, tokenizer)
    kwargs = {
        'num_workers': 0,
        'pin_memory': True
    }

    eval_loader = DataLoader(eval_dataset,
                             batch_size=4,
                             collate_fn=eval_dataset.fn_collate,
                             shuffle=False,
                             **kwargs)

    char_total_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
    word_total_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
    for batch in tqdm(eval_loader, ascii=True, desc='Evaluation'):
        batch_errors = compute_perfs(batch, model, preprocess, idx2token, blank_idx, device)
        char_total_errors.update(batch_errors[0])
        word_total_errors.update(batch_errors[1])

    print('Char errors..........')
    for editop, _ in char_total_errors.items():
        char_total_errors[editop] /= len(eval_dataset)
    print(char_total_errors)

    print('Word errors..........')
    for editop, _ in word_total_errors.items():
        word_total_errors[editop] /= len(eval_dataset)
    print(word_total_errors)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script to train attention models.')
    parser.add_argument('--config', default='',
                        help='The config file used for training the model.')
    parser.add_argument('--checkpoint', default='',
                        help='The checkpoint model to be evaluated.')
    parser.add_argument('--eval_data', default='',
                        help='The eval data.')
    args = parser.parse_args()

    config_file = args.config
    try:
        options = OmegaConf.load(config_file)
    except  Exception as e:
        logging.error('Error reading configuration file: %s', e)

    main(options, args.eval_data, args.checkpoint)
