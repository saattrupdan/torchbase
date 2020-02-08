import torch
from pathlib import Path
from .typing import *

def parametrised(decorator: Decorator) -> Decorator:
    ''' A meta-decorator that enables parameters in inner decorator. '''
    def parametrised_decorator(*args, **kwargs):
        if isinstance(args[0], type):
            return decorator(*args, **kwargs)
        else:
            def repl(cls: type):
                return decorator(cls, *args, **kwargs)
            return repl
    return parametrised_decorator

@parametrised
def changename(f: Function, name: str) -> Function:
    f.__name__ = name
    return f

@parametrised
def magic(cls: type,
    criterion: Criterionlike = 'binary_cross_entropy',
    optimiser: Optimiserlike = 'adamw',
    scheduler: nSchedulerlike = 'reduce_on_plateau',
    metrics: MetriclikesOrString = [],
    data_dir: Pathlike = '.data',
    verbose: int = 1,
    tensorboard: bool = False,
    monitor: nStr = None,
    minimise_monitor: nBool = None,
    target_value: nFloat = None,
    patience: nNumeric = 9,
    smoothing: float = 0.99,
    save_model: bool = True,
    overwrite: bool = True,
    learning_rate: float = 3e-4) -> type:
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
        _tensorboard = tensorboard
        _monitor = monitor
        _minimise_monitor = minimise_monitor
        _target_value = target_value
        _patience = patience
        _smoothing = smoothing
        _save_model = save_model
        _overwrite = overwrite
        _learning_rate = learning_rate 

    return MagicModule
