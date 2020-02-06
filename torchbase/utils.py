import torch
import torch.optim.lr_scheduler as sched
from .decorators import changename
from .typing import *
from .metrics import *

def str2optim(optimiser: Optimiserlike, model: Module) -> Optimiser:
    if not isinstance(optimiser, str):
        return optimiser
    elif optimiser == 'adam':
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

def str2sched(scheduler: str, optimiser: Optimiser) -> Scheduler:
    if not isinstance(scheduler, str):
        return scheduler
    elif scheduler == 'reduce_on_plateau':
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
        raise RuntimeError(f'Scheduler {scheduler} not found.')

def str2crit(criterion: Metriclike) -> Metric:
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
        criterion.__name__ = 'loss'
        return criterion

def str2metric(metric: Metriclike, wrapper: Wrapper) -> Metric:
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
