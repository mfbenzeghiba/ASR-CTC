"""Define a single custumized model."""

import os
from typing import List
from collections import Counter
import logging
import json

from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from utils.metrics import greedy_decoding, compute_cer,  IterMeter
from utils.optimizers import set_optimizer
from utils.schedulers import set_scheduler
from utils.models_utils import load_checkpoint, create_model

from data_transforms.processing import create_processing


class CTCASR():
    """Class that define a custom model.

    Args:
        ASRBASEModel (_type_): _description_
    """

    def __init__(self,
                 cfg: dict,
                 exp_folder: str,
                 phases: List[str],
                 device: torch.device='cpu') -> None:
        super(CTCASR, self).__init__()

        self.exp_folder = exp_folder
        self.device = device
        self.cfg = cfg
        self.model_folder = cfg.folder
        self.model_name = cfg.name
        self.num_labels = cfg.num_labels

        self._learning_rate = None
        self._optimizer = None
        self._scheduler = None

        self.epoch = 0
        self.best_error = float('inf')
        self.best_epoch = 0

        self.processing = None

        # probably we need to check the pretrained model
        self.pretrained_model = cfg.get('from_pretrained', None)

        self.train_state = []
        self.train_meter = {}
        for phase in phases:
            self.train_meter[phase] = IterMeter()

        self.checkpoint_dir = os.path.join(self.exp_folder, self.cfg.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            raise EOFError(f'{self.checkpoint_dir} already exists')
        logging.info('Model folder: %s', self.checkpoint_dir)
        
        print(f'Create the model: {self.model_name}')
        self.model = create_model(self.cfg)
        self.save_checkpoint(epoch=0)
        self.model.to(self.device)        

        if self.cfg.processing and self.cfg.processing is not None:
            proc = self.cfg.processing
            self.processing = create_processing(proc.name, **proc.options)
        logging.info( 'Data processing: %s', self.processing )

        if self._learning_rate is None:
            self.set_learning_rate()
        if self._optimizer is None:
            self.set_optimizer()
        if self._scheduler is None:
            self.set_scheduler()


    @property
    def parameters(self):
        """Return parameters to be updated."""

        parameters = []
        for param in self.model.parameters():
            if param.requires_grad:
                parameters.append(param)
        return parameters

    def freeze_parameters(self, base: bool=True):
        """Freeze model parameters."""

        self.model.freeze_parameters(base)

    @property
    def phase(self):
        """Return the current phase."""

        return self._phase


    @phase.setter
    def phase(self, phase: str=None) -> None:
        """Set the phase."""

        self._phase = phase
        if self._phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        self._reset_metrics()


    @property
    def learning_rate(self):
        """return the learning rate."""

        return self._learning_rate


    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set the learning rate."""

        self._learning_rate = value


    def set_learning_rate(self):
        """Set the learning rate"""

        lr = self.cfg.get('learning_rate', None)
        if lr is not None:
            lr_type = 'provided'
        else:
            lr = 0.001
            lr_type = 'default'

        print(f'Set the {lr_type} learning rate: {lr}')
        self.learning_rate = lr


    @property
    def optimizer(self) -> Optimizer:
        """Set the optimizer."""

        return self._optimizer


    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        """Set the optimizer

        Args:
            optimizer (Optimizer): The optimizer
        """

        self._optimizer = value


    def set_optimizer(self):
        """Set the optimizer"""

        if self.cfg.optimizer and self.cfg.optimizer.name is not None:
            opname = self.cfg.optimizer.name
            kwargs = self.cfg.optimizer.get('options', None)
            kwargs.lr = self.learning_rate
            optimizer = set_optimizer(self, opname, **kwargs)
            opt_type = 'provided'
        else:
            opname = 'Adam'
            optimizer = set_optimizer(self, opname)
            opt_type = 'default'

        print(f'Set the {opt_type} optimizer: {optimizer}')
        self.optimizer = optimizer


    @property
    def scheduler(self) -> Optimizer:
        """Return the scheduler."""

        return self._scheduler


    @scheduler.setter
    def scheduler(self, value: Optimizer = 'ReduceLROnPlateau') -> None:
        """Set the scheduler given in the config file or set the default one.

        Args:
            scheduler (Optimizer, optional): _description_. Defaults to 'ReduceLROnPlateau'.
        """

        self._scheduler = value


    def set_scheduler(self) -> None:
        """Set the scheduler."""

        if self.cfg.scheduler and self.cfg.scheduler.name is not None:
            schname = self.cfg.scheduler.name
            kwargs = self.cfg.scheduler.get('options', None)
            scheduler = set_scheduler(self.optimizer, schname, **kwargs)
            sch_type = 'provided'
        else:
            schname = 'ReduceLROnPlateau'
            scheduler = set_scheduler(self.optimizer, schname)
            sch_type = 'default'

        print(f'Set the {sch_type} scheduler {scheduler}')
        self.scheduler = scheduler


    def is_better(self, epoch:int) -> bool:
        """Compare the current model to the previous best model.

        Args:
            epoch (int): Epoch number

        Returns:
            bool: True/False
        """

        epoch_error = self.train_meter[self.phase].epoch_error()
        if epoch_error  < self.best_error:
            self.best_error = epoch_error
            self.best_epoch = epoch
            return True
        return False


    def trainer_state_update(self, epoch:int) -> None:
        """Update the training status.

        Args:
            epoch (int): Epoch number
        """

        current_loss, current_errors = self.train_meter[self.phase].current_metrics()
        self.train_state.append(
            {
                'epoch': epoch,
                f'{self.phase}_loss': f'{current_loss:.3f}',
                f'{self.phase}_sub': f'{current_errors["replace"]:.3f}',
                f'{self.phase}_del': f'{current_errors["delete"]:.3f}',
                f'{self.phase}_ins': f'{current_errors["insert"]:.3f}',
                f'{self.phase}_cer': f'{current_errors["cer"]:.3f}',
                'learning rate': self.optimizer.param_groups[0]['lr']
            }
        )


    def step_update(self, nb_samples: int, loss: float, errors: float) -> None:
        """Update the statistics after each step.

        Args:
            nb_samples (int): number of samples in a batch
            loss (float): the loss value
            errors (float): the edition errors
        """

        self.train_meter[self.phase].update_step_metric(nb_samples, loss, errors)

    def _reset_metrics(self):
        """Reset metrics."""

        self.train_meter[self.phase].reset()

    def run_step(self, batch: torch.Tensor, criterion: torch.nn, idx2token: dict, blank_idx: int = 0):
        """Train and validate the model on the datasets.

        Args:
            batch (torch.Tensor): _description_
        """

        _, waveforms, waveform_sizes, targets, target_sizes = batch

        targets = targets.to(self.device)
        target_sizes = target_sizes.to(self.device)

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(self.phase == 'train'):
            logits, input_size = self.model.process_batch(
                waveforms=waveforms,
                waveform_sizes=waveform_sizes,
                processing=self.processing,
                device=self.device
            )
            log_probs = F.log_softmax(logits, dim=2) # (T, B, C)
            loss = criterion(log_probs.transpose(1, 0), targets, input_size, target_sizes)

            if self.phase == 'train':
                loss.backward()
                self._optimizer.step()

        _, hyps = greedy_decoding(log_probs.cpu(), input_size.cpu(),
                                  idx2token, blank_idx)
        refs = []
        for i, _ in enumerate(targets):
            seq_labels = targets[i][:target_sizes[i]].tolist()
            refs.append([idx2token[k] for _, k in enumerate(seq_labels)])
        errors = compute_cer(refs, hyps)
        nb_samples = input_size.shape[0]
        step_loss = loss.detach().cpu().item()

        self.step_update(nb_samples, step_loss, errors)

    def write_train_summary(self, train_log_file: str) -> None:
        """Write to a file the training state."""

        with open(train_log_file, "w", encoding='utf-8') as fout:
            json.dump(self.train_state, fout, indent=4)


    def save_checkpoint(self, epoch: int, epoch_loss: float=None, train_error: Counter=None,
                        valid_error: Counter=None) -> None:
        """Save the model parameters.

        Args:
            output_folder (string): Path to save the checkpoint.
            epoch (integer): Epoch number.
            epoch_loss: (float): The epoch loss.
            train_error (Counter): Edition Errors rate on the train dataset.
            valid_error (Counter): Edition Errors rate on the validation dataset.
            if_best_only (bool): save the model only if it is better than the previous one.
        """

        package = {
                   'epoch': epoch,
                   'num_classes': self.num_labels,
                   'optimizer': self.optimizer,
                   'scheduler': self.scheduler,
                   'epoch_loss': epoch_loss,
                   'train_error': train_error,
                   'valid_error': valid_error,
                   'state_dict': self.model.state_dict()
        }
        model_cfg = OmegaConf.create(OmegaConf.to_yaml(self.cfg, resolve=True))
        package['cfg'] = model_cfg
        file_name = f'checkpoint_epoch{epoch}.pt'
        checkpoint = os.path.join(self.checkpoint_dir, file_name)
        torch.save(package, checkpoint)
