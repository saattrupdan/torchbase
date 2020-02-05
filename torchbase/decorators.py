import torch
from pathlib import Path
from .modules import ModuleWrapper
from .utils import Metric
from .typing import *

def parametrised(decorator: Decorator) -> Decorator:
    ''' A meta-decorator that enables parameters in inner decorator. '''
    def parametrised_decorator(*args, **kwargs):
        def repl(cls: Module):
            return decorator(cls, *args, **kwargs)
        return repl
    return parametrised_decorator

@parametrised
def magic(cls: Module,
    criterion: Metriclike = 'binary_cross_entropy',
    optimiser: Optimiserlike = 'adamw',
    scheduler: nSchedulerlike = 'reduce_on_plateau',
    data_dir: Pathlike = '.',
    verbose: int = 1) -> Wrapper:
    ''' Adds more functionality to a PyTorch Module. '''

    class MagicModule(ModuleWrapper):
        _model_class = cls
        _optimiser = optimiser
        _scheduler = scheduler
        _criterion = criterion
        _data_dir = data_dir
        _verbose = verbose

    return MagicModule

#def inheritdoc(func):
#    
