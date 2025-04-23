"""Useful metrics to evaluate the performance of the models."""

from collections import Counter
from typing import List, Tuple
import Levenshtein as Lev

import torch


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.nb_steps = 0
        self.loss = 0.
        self.nb_samples = 0
        self.current_loss = 0.
        self.errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
        self.current_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})

    def update_step_metric(self, n: int, step_loss: float, step_errors: Counter=None) -> None:
        """Update the metric.

        Args:
            n (int): Number of samples
            step_loss (flout): The loss value
            step_errors (Counter, optional): The edit distance
        """

        self.nb_steps += 1
        self.nb_samples += n
        self.loss += step_loss
        if step_errors is not None:
            self.errors.update(step_errors)

    def get_metrics(self) -> Tuple[int, int, Counter]:
        """Return the current metrics."""

        avg_loss = self.loss / self.nb_steps
        avg_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
        for editop, value in self.errors.items():
            avg_errors[editop] = value/self.nb_samples

        return self.nb_steps, avg_loss, avg_errors

    def reset(self) -> None:
        """Reset the variables."""

        self.errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
        self.current_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
        self.loss = 0.
        self.current_loss = 0.
        self.nb_samples = 0
        self.nb_steps = 0

    def current_metrics(self) -> Tuple[int, Counter]:
        """Return the current metrics."""

        self.current_loss = self.loss/self.nb_steps
        for editop, _ in self.errors.items():
            self.current_errors[editop] = self.errors[editop]/self.nb_samples
        return self.current_loss, self.current_errors

    def epoch_error(self) -> float:
        """get the current token error."""

        return self.current_errors['cer']

    @property
    def steps(self) -> int:
        """return the total number of steps."""

        return self.nb_steps

    @property
    def samples(self) -> int:
        """Return number of samples."""

        return self.nb_samples

def greedy_decoding(preds: List[torch.Tensor], input_size: List[int],
                    idx2token: dict, blank_idx: int = 0) -> Tuple[List[int], List[str]]:
    """Evaluate the model perfomance.

    This function evaluates the performance of the model trained with CTC and
    returns the output string. It uses the Levenstein edit distance to get the output value.
    It outputs the token error rate.

    Args:
        preds (List[torch.Tensor]): The prediction of the model.
        input_size (List[torch.IntTensor]): The length of the target sequence.
        idx2char (dict): index to token dictionary.
        blank_idx (int, optional): Index of the blank symbol.
    """

    arg_maxes = torch.argmax(preds, dim=2)
    str_sequences = []
    idx_sequences = []
    for i, args in enumerate(arg_maxes):
        decode = []
        prev_token = blank_idx
        for k, token in enumerate(args[:input_size[i]]):
            if token.item() != blank_idx:
                if token.item() != prev_token:
                    decode.append(token.item())
                    prev_token = token.item()
                else:
                    continue
            else:
                prev_token = blank_idx
                continue
        idx_sequences.append(decode)
        str_sequences.append([idx2token[k] for _, k in enumerate(decode)])
    return idx_sequences, str_sequences


def compute_cer(refs: List[str], hyps: List[str]) -> Counter:
    """Compute the character error rate and return the edit distances.

    Args:
        hyps (List[str]): The hypotheses text
        refs (List[str]): The reference text
    """
    batch_cer = {'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0}

    for ref, hyp in zip(refs, hyps):
        errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'cer': 0})
        errors.update(([e[0] for e in Lev.editops(ref, hyp)]))

        errors['cer'] = sum(errors.values())
        for editop, _ in batch_cer.items():
            batch_cer[editop] += errors[editop]/len(ref)
    return batch_cer

if __name__ == '__main__':

    str1 = [['a', 'b', 'c','r']]
    str2 = [['a', 'b', 'dc']]
    str1 = [['h#', 'w', 'iy', 'h#', 'k', 'ih', 'ng', 'h#', 'g', 'eh', 'dx', 'ih', 'h#']]
    str2 = [['h#', 'ih', 'h#', 'ih', 'n', 'h#', 'k', 'ih', 'ih', 'b', 'ih', 'h#']]
    str1 = [['je', 'ne', 'suis']]
    str2 = [['je', 'ne', 'suis', 'pas']]

    err = compute_cer(str1, str2)
    print(err)
