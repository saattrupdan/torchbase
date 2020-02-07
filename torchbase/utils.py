import torch
import torch.optim.adam as adam
import torch.optim.adadelta as adadelta
import torch.optim.adagrad as adagrad 
import torch.optim.adamw as adamw
import torch.optim.sparse_adam as sparse_adam
import torch.optim.adamax as adamax
import torch.optim.rmsprop as rmsprop
import torch.optim.sgd as sgd
import torch.optim.lr_scheduler as sched
from .decorators import changename
from .typing import *
from .metrics import (accuracy, accuracy_with_logits, 
                      samples_f1, samples_f1_with_logits)

def str2optim(optimiser: Optimiserlike, model: Module) -> Optimiser:
    if not isinstance(optimiser, str):
        return optimiser
    elif optimiser == 'adam':
        return adam.Adam(model.parameters())
    elif optimiser == 'adadelta':
        return adadelta.Adadelta(model.parameters())
    elif optimiser == 'adagrad':
        return adagrad.Adagrad(model.parameters())
    elif optimiser == 'adamw':
        return adamw.AdamW(model.parameters())
    elif optimiser == 'sparse_adam':
        return sparse_adam.SparseAdam(model.parameters())
    elif optimiser == 'adamax':
        return adamax.Adamax(model.parameters())
    elif optimiser == 'rmsprop':
        return rmsprop.RMSprop(model.parameters())
    elif optimiser == 'sgd':
        return sgd.SGD(model.parameters(), lr = 3e-4)
    else:
        raise RuntimeError(f'Optimiser {optimiser} not found.')

def str2sched(scheduler: Schedulerlike, optimiser: Optimiser,
    dataloader: DataLoader, epochs: Numeric, patience: Numeric) -> Scheduler:
    if not isinstance(scheduler, str):
        return scheduler
    elif scheduler == 'reduce_on_plateau':
        if not isinstance(patience, int): patience = 20
        return sched.ReduceLROnPlateau(optimiser, patience = patience // 2)
    elif scheduler == 'cyclic':
        return sched.CyclicLR(optimiser, base_lr = 1e-4, max_lr = 1.)
    elif scheduler == 'step':
        return sched.StepLR(optimiser, step_size = 5)
    elif scheduler == 'exp':
        return sched.ExponentialLR(optimiser, gamma = 0.1)
    else:
        raise RuntimeError(f'Scheduler {scheduler} not found.')

def str2crit(criterion: Criterionlike) -> Criterion:
    if not isinstance(criterion, str):
        return criterion
    else:
        if criterion == 'mean_absolute_error':
            criterion = torch.nn.L1Loss()
        elif criterion == 'mean_squared_error':
            criterion = torch.nn.MSELoss()
        elif criterion == 'binary_cross_entropy':
            criterion = torch.nn.BCELoss()
        elif criterion == 'categorical_cross_entropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion == 'neg_log_likelihood':
            criterion = torch.nn.NLLLoss()
        elif criterion == 'binary_cross_entropy_with_logits':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion == 'ctc':
            criterion = torch.nn.CTCLoss()
        else:
            raise RuntimeError(f'Criterion {criterion} not found.')
        type(criterion).__name__ = 'loss'
        return criterion

def str2function(metric: Functionlike, wrapper: Wrapper) -> Function:
    if not isinstance(metric, str):
        return metric
    else:
        puremetric = metric.replace('val_', '')
        if puremetric == 'loss':
            return wrapper.criterion
        elif puremetric == 'accuracy':
            return accuracy
        elif puremetric == 'accuracy_with_logits':
            return accuracy_with_logits
        elif puremetric == 'samples_f1':
            return samples_f1
        elif puremetric == 'samples_f1_with_logits':
            return samples_f1_with_logits
        else:
            raise RuntimeError(f'Metric {metric} not found.')
