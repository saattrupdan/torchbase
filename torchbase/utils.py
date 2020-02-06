import torch
import torch.optim.lr_scheduler as sched
from .typing import *
from .metrics import accuracy, samples_f1

def str2optim(optimiser: Optimiserlike, model: Module) -> Optimiser:
    if isinstance(optimiser, Optimiser):
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

def str2crit(criterion: Metriclike) -> Metric:
    if isinstance(criterion, Metric):
        return criterion
    elif criterion == 'mean_absolute_error':
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
    if isinstance(scheduler, Scheduler):
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

def str2metric(metric: Metriclike, wrapper: Wrapper) -> Metric:
    if isinstance(metric, Metric):
        return metric
    else:
        puremetric = metric.replace('val_', '')
        if puremetric == 'loss':
            return wrapper.criterion
        elif puremetric == 'accuracy':
            return accuracy
        elif puremetric == 'samples_f1':
            return samples_f1
        else:
            raise RuntimeError(f'Metric {metric} not found.')

def getname(thing: object):
    if isinstance(thing, str): 
        return thing
    else: 
        return thing.__name__
