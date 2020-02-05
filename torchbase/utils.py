import torch
import torch.optim.lr_scheduler as sched
from .typing import *

def str2optim(optimiser: str, model: Module) -> Optimiser:
    if optimiser == 'adam':
        return torch.optim.Adam(model.parameters())
    elif optimiser == 'adadelta':
        return torch.optim.AdaDelta(model.parameters())
    elif optimiser == 'adagrad':
        return torch.optim.AdaGrad(model.parameters())
    elif optimiser == 'adamw':
        return torch.optim.AdamW(model.parameters())
    elif optimiser == 'sparse_adam':
        return torch.optim.SparseAdam(model.parameters())
    elif optimiser == 'adamax':
        return torch.optim.Adamax(model.parameters())
    elif optimiser == 'rmsprop':
        return torch.optim.RMSProp(model.parameters())
    elif optimiser == 'sgd':
        return torch.optim.SGD(model.parameters())
    else:
        raise RuntimeError(f'Optimiser {optimiser} not found.')

def str2crit(criterion: str) -> Metric:
    if criterion == 'mean_absolute_error':
        return torch.nn.L1Loss()
    elif criterion == 'mean_squared_error':
        return torch.nn.MSELoss()
    elif criterion == 'binary_cross_entropy':
        return torch.nn.BCELoss()
    elif criterion == 'categorical_cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif criterion == 'neg_log_likelihood':
        return torch.nn.NLLLoss()
    elif criterion == 'binary_cross_entropy_with_logits':
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == 'ctc':
        return torch.nn.CTCLoss()
    else:
        raise RuntimeError(f'Criterion {criterion} not found.')

def str2sched(scheduler: str, optimiser: Optimiser) -> Scheduler:
    if scheduler == 'reduce_on_plateau':
        return sched.ReduceLROnPlateau(optimiser)
    elif scheduler == 'one_cycle':
        return sched.OneCycleLR(optimiser, max_lr = 1.)
    elif scheduler == 'cyclic':
        return sched.CyclicLR(optimiser, base_lr = 1e-4, max_lr = 1.)
    elif scheduler == 'step':
        return sched.StepLR(optimiser, step_size = 5)
    elif scheduler == 'exp':
        return sched.ExponentialLR(optimiser, gamma = 0.1)
    else:
        raise RuntimeError(f'Scheduler {self.scheduler} not found.')
