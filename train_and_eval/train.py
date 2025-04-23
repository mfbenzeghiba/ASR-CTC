"""Script to train a speech recognition systems."""

import os
import sys
from pathlib import Path
import json
import argparse
import logging
import shutil
import time

from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from models.ctc_asr import CTCASR
from data_providers.timit.timit import TimitTokenizer, TimitDataset


def main(cfg_path: str, options: OmegaConf):
    """Train the model with the given options.

    Args:
        cfg_path (str): path to the YAML config file
        options (dict): Dictionary with the training options
    """

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    common_options = options.common
    data_options = options.datasets
    exp_folder = options.exp_dir

    # save the config file in the exp folder
    shutil.copy(cfg_path, exp_folder)

    seed = common_options.seed
    torch.manual_seed(seed)

    use_cuda = common_options.use_cuda
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # Create the tokenizer
    tokenizer = TimitTokenizer(data_options)
    num_classes = len(tokenizer)
    print(num_classes)
    print(tokenizer.index_to_token)

    train_dataset = TimitDataset(data_options.train, tokenizer)
    valid_dataset = TimitDataset(data_options.valid, tokenizer)
    kwargs = {}
    if use_cuda:
        kwargs = {
            'num_workers': data_options.num_workers,
            'pin_memory': True
        }

    train_loader = DataLoader(train_dataset,
                              batch_size=data_options.batch_size,
                              collate_fn=train_dataset.fn_collate,
                              shuffle=data_options.shuffle_train,
                              **kwargs)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=data_options.batch_size,
                              collate_fn=valid_dataset.fn_collate,
                              shuffle=False,
                              **kwargs)

    logging.info("-" * 50)
    logging.info('Database: %s', data_options.database)
    logging.info('Train data: %s', data_options.train)
    logging.info('Valid data: %s', data_options.valid)
    logging.info('Batch size: %s', data_options.batch_size)
    logging.info('Number of training batch: %d', len(train_loader))
    logging.info('Number of validation batch: %d', len(valid_loader))
    logging.info('Blank token: %s', tokenizer.blank_token)
    logging.info('Blank index: %d', tokenizer.blank_idx)
    logging.info('Experiment folder: %s', exp_folder)
    logging.info('PyTorch Version: %s', torch.__version__)
    logging.info('Device: %s', device)
    logging.info("-" * 50)

    # Create the model
    model_options = options.model
    model = CTCASR(cfg=model_options, exp_folder=exp_folder,
                   phases=common_options.phases, device=device)
    total_params = sum( param.numel() for param in model.parameters )
    print(f'Number of adapted parameters: {total_params}')
    print(model.model)

    
    criterion = torch.nn.CTCLoss(zero_infinity=True, reduction='mean', blank=tokenizer.blank_idx).to(device)
    phases = common_options.phases
    bepoch = model.epoch
    nepochs = common_options.nepochs
    begin_train_time = time.time()

    training_log_file = os.path.join(exp_folder, 'training_status.json')

    # Start training
    print(bepoch)
    update_threshold = model_options.freeze_finetune_updates * nepochs

    for epoch in range(bepoch+1, nepochs):
        logging.info('Epoch --- %d', epoch)
        print('scheduler ', model.scheduler.get_last_lr())
        print('optimizer ', model.optimizer.param_groups[0]['lr'])

        if epoch < update_threshold:
            model.freeze_parameters(base=True)
        else:
            model.freeze_parameters(base=False)

        for _, phase in enumerate(phases):
            logging.info('%s phase....', phase)
            model.phase = phase
            if phase == 'train':
                phase_loader = train_loader
            else:
                phase_loader = valid_loader
            for batch in tqdm(phase_loader, ascii=True, desc=phase.upper()):
                model.run_step(
                    batch=batch,
                    criterion=criterion,
                    idx2token=tokenizer.index_to_token,
                    blank_idx=tokenizer.blank_idx
                )
            epoch_loss, epoch_errors = model.train_meter[phase].current_metrics()
            logging.info('Model: %s\t loss: %.3f\t token error: %.3f',
                             model.model_name, epoch_loss, epoch_errors['cer'])
            if phase == 'valid':
                if model.is_better(epoch):
                    model.save_checkpoint(epoch, num_classes, epoch_loss, epoch_errors)
                if model.scheduler is not None:
                    val_loss = round(epoch_loss, 3)
                    model.scheduler.step(val_loss)
                    
            model.trainer_state_update(epoch)
        model.write_train_summary(training_log_file)

    end_train_time = time.time()
    train_time = (end_train_time - begin_train_time) / 3600

    train_summary = {
        'model_name': model.model_name,
        'best_checkpoint': os.path.join(model.checkpoint_dir, 'checkpoint_best.pth'),
        'best_epoch': model.best_epoch,
        'best_error': f'{model.best_error:.3f}',
        'learning_rate': model.learning_rate,
        'batch_size': data_options.batch_size,
        'nb_batches': len(train_loader),
        'training_time': f'{train_time:.3f}'
    }

    model.train_state.insert(0, train_summary)
    with open(training_log_file, "w", encoding='utf-8') as fout:
        json.dump(model.train_state, fout, indent=4)
    print(model.train_state)

    logging.info('Training lasts %.0fm%.0f', train_time//60, train_time%60)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script to train attention models.')
    parser.add_argument('--config', default='conf/ctc_config.yaml',
                        help='conf file with argument of LSTM and training')
    args = parser.parse_args()

    config_path = args.config
    print(config_path)
    try:
        opts = OmegaConf.load(config_path)
        print(opts)
    except  Exception as e:
        logging.error('Error reading configuration file: %s',  e)

    exp_dir = opts.exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    main(config_path, opts)
