import torch
import logging
from pathlib import Path
from .training import train_model
from .plotting import plot
from .utils import str2optim, str2crit
from .io import save, save_model, load
from .typing import *

class ModuleWrapper:
    ''' A PyTorch module with logging and training built-in.

    INPUT
        model_class: class
            A subclass of torch.nn.Module
        criterion: Metric or str = 'binary_cross_entropy'
            The loss function used for training
        optimiser: Optimiser, str or None
            The optimiser used for training
        scheduler: Scheduler, str or None
            The optimiser used for training
        data_dir: Path or str = '.'
            The data directory
        verbose: {0, 1, 2} = 1
            Verbosity level
    '''

    _model_class: nType = None
    _optimiser: nOptimiserlike = None
    _scheduler: nSchedulerlike = None
    _criterion: nCriterionlike = None
    _metrics: nMetriclikes = None
    _data_dir: nPathlike = None
    _verbose: nInt = None
    _tensorboard: nBool = None
    _monitor: nStr = None
    _minimise_monitor: nBool = None
    _target_value: nFloat = None
    _patience: nNumeric = None
    _smoothing: nFloat = None
    _save_model: nBool = None
    _overwrite: nBool = None
    _learning_rate: nFloat = None

    # Fetch methods from other modules
    fit: Function = train_model
    save: Function = save
    save_model: Function = save_model
    plot: Function = plot
    load: Function = classmethod(load)

    def __init__(self, *args, **kwargs):
        from .metrics import MetricObject

        self.model: Wrapper = type(self)._model_class(*args, **kwargs)
        self.model_name: str = type(self)._model_class.__name__
        self.optimiser: Optimiserlike = type(self)._optimiser
        self.scheduler: Schedulerlike = type(self)._scheduler
        self.criterion: Criterionlike = type(self)._criterion
        self.monitor: str = type(self)._monitor
        self.minimise_monitor: nBool = type(self)._minimise_monitor
        self.target_value: nFloat = type(self)._target_value
        self.patience: nNumeric = type(self)._patience
        self.smoothing: float = type(self)._smoothing
        self.save_model: bool = type(self)._save_model
        self.overwrite: bool = type(self)._overwrite
        self.tensorboard: bool = type(self)._tensorboard
        self.data_dir: Path = Path(str(type(self)._data_dir))
        self.verbose: int = type(self)._verbose
        self.history: History = {'loss': [], 'val_loss': []}

        # Initialise criterion, optimiser and scheduler
        if isinstance(self.criterion, str):
            self.criterion = str2crit(self.criterion)
        if isinstance(self.optimiser, str):
            self.optimiser = str2optim(self.optimiser, self.model,
                lr = type(self)._learning_rate)

        # Initialise metrics
        if not isinstance(type(self)._metrics, list):
            unique_metrics: set = {type(self)._metrics}
        else:
            unique_metrics: set = set(type(self)._metrics)
        unique_metrics.add('loss')
        unique_metrics = {metric for metric in unique_metrics 
                          if metric[:4] != 'val_'}
        sorted_metrics: list = ['loss'] + sorted(unique_metrics - {'loss'})
        self.metrics: Metrics = [MetricObject(metric, self, val = False) 
                        for metric in sorted_metrics]
        self.metrics += [MetricObject(metric, self, val = True) 
                         for metric in sorted_metrics]

        # Set up logging
        logging_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging_levels[self.verbose])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        handler.setLevel(logging_levels[self.verbose])
        self.logger.addHandler(handler)

    def trainable_params(self) -> int:
        ''' Returns the number of trainable parameters in the model. '''
        return sum(param.numel() for param in self.model.parameters() 
                if param.requires_grad)

    def is_cuda(self) -> bool:
        return next(self.model.parameters()).is_cuda

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        return self.model.__repr__(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        return self.model.cpu(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
