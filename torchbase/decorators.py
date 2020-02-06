import torch
from pathlib import Path
from .typing import *

def parametrised(decorator: Decorator) -> Decorator:
    ''' A meta-decorator that enables parameters in inner decorator. '''
    def parametrised_decorator(*args, **kwargs):
        def repl(cls: Module):
            return decorator(cls, *args, **kwargs)
        return repl
    return parametrised_decorator

@parametrised
def changename(f: Function, name: str) -> Function:
    f.__name__ = name
    return f

@parametrised
def magic(cls: Module,
    criterion: Metriclike = 'binary_cross_entropy',
    optimiser: Optimiserlike = 'adamw',
    scheduler: nSchedulerlike = 'reduce_on_plateau',
    metrics: MetriclikesOrString = [],
    data_dir: Pathlike = '.data',
    verbose: int = 1) -> Wrapper:
    ''' Adds more functionality to a PyTorch Module. '''
    from .modules import ModuleWrapper

    class MagicModule(ModuleWrapper):
        _model_class = cls
        _optimiser = optimiser
        _scheduler = scheduler
        _criterion = criterion
        _metrics = metrics
        _data_dir = data_dir
        _verbose = verbose

    return MagicModule
