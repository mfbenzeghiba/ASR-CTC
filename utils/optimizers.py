"""classes and method to help define the optimizer.
https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/torchensemble/utils/set_module.py
"""

import importlib

def set_optimizer(model, optimizer_name, **kwargs):
    """
    Set the parameter optimizer for the model.

    Reference: https://pytorch.org/docs/stable/optim.html#algorithms
    """

    supported_optimizers = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "RMSprop",
        "SGD",
    ]

    if optimizer_name not in supported_optimizers:
        raise NotImplementedError(
            f'Unrecognized optimizer: {optimizer_name}, \
            should be one of {",".join(supported_optimizers)}.'
        )

    optimizer_cls = getattr(
        importlib.import_module("torch.optim"), optimizer_name
    )
    optimizer = optimizer_cls(model.parameters, **kwargs)

    return optimizer
