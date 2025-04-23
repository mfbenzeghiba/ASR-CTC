"""Classes and functions to help schedule the learning rate.
https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/torchensemble/utils/set_module.py
"""

import importlib
from typing import Optional
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    """Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


class LinearCosineAnnealingLR(LearningRateScheduler):
    def __init__(
            self,
            optimizer: optim.Optimizer,
            lr: float,
            warmup_steps: int,
            t_max: int
    ) -> None:
        """Define the LinearCosineAnnealingL scheduler.

        Args:
            optimizer (optim.Optimizer): The optimizer
            lr (float): The initial learning rate
            warmup_steps (int): number of warmup steps
            t_max (int): Maximum number of iterations.
        """

        super(LinearCosineAnnealingLR, self).__init__(optimizer, lr)
        self.current_step = 0
        self.optimizer = optimizer
        self.nsteps = warmup_steps + t_max
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        lr = self.lr / self.warmup_steps
        self.set_lr(self.optimizer, lr)

    def step(self, val_loss: Optional[torch.FloatTensor] = None) -> None:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.lr * (self.current_step+1)/self.warmup_steps
            self.set_lr(self.optimizer, lr)
        elif self.current_step < self.nsteps:
            self.scheduler.step()
        else:
            pass

    def get_last_lr(self):
        return self.get_lr()

# see:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py

class TriStageLRScheduler(LearningRateScheduler):
    """Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        init_lr_scale (float): Initial learning rate scale.
        final_lr_scale (float): Final learning rate scale.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        hold_steps (int): Hold the learning rate for the N updates.
        decay_steps (int): Decay the learning rate linearly for the first N updates.
        total_steps (int): Total steps in training.
    """
    def __init__(
            self,
            optimizer: optim.Optimizer,
            lr: float,
            init_lr_scale: float,
            final_lr_scale: float,
            warmup_ratio: float,
            hold_ratio: float,
            decay_ratio: float,
            max_updates: int,
    ) -> None:
        
        assert max_updates > 0
        sum_ratio = warmup_ratio + hold_ratio + decay_ratio
        assert sum_ratio == 1, "phase ratios must add up to 1"

        super(TriStageLRScheduler, self).__init__(optimizer, lr)
        self.peak_lr = lr
        self.init_lr = init_lr_scale * lr
        self.final_lr = final_lr_scale * lr

        self.warmup_steps = int(max_updates * warmup_ratio)
        self.hold_steps = int(max_updates * hold_ratio)
        self.decay_steps = int(max_updates * decay_ratio)
        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify phase ratios"

        if self.warmup_steps != 0:
            self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps
        else:
            self.warmup_rate = 0

        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr

    def get_last_lr(self):
        return self.get_lr()


def set_scheduler(optimizer: optim.Optimizer, scheduler_name: str, **kwargs) -> optim.lr_scheduler:
    """Set the scheduler

    Args:
        optimizer (optim.Optimizer): The optimizer
        scheduler_name (str): The scheduler name

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    supported_lr_schedulers = [
        'LambdaLR',
        'MultiplicativeLR',
        'StepLR',
        'MultiStepLR',
        'ExponentialLR',
        'CosineAnnealingLR',
        'ReduceLROnPlateau',
        'CyclicLR',
        'OneCycleLR',
        'CosineAnnealingWarmRestarts',
        'LinearCosineAnnealingLR',
        'TriStageLRScheduler'
    ]

    if scheduler_name not in supported_lr_schedulers:
        raise NotImplementedError(
            f'Unrecognized scheduler: {scheduler_name}, \
                should be one of {",".join(supported_lr_schedulers)}.'
        )
    if scheduler_name == 'LinearCosineAnnealingLR':
        scheduler = LinearCosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == 'TriStageLRScheduler':
        scheduler = TriStageLRScheduler(optimizer, **kwargs)
    else:
        scheduler_cls = getattr(
            importlib.import_module("torch.optim.lr_scheduler"), scheduler_name
            )
        scheduler = scheduler_cls(optimizer, **kwargs)

    return scheduler
