import torch
import logging
from pathlib import Path
from functools import cmp_to_key
from .training import train_model
from .plotting import plot
from .utils import str2crit, str2optim, str2sched
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

    _model_class = None
    _optimiser = None
    _scheduler = None
    _criterion = None
    _data_dir = None
    _verbose = None

    # Fetch methods from other modules
    fit = train_model
    save = save
    save_model = save_model
    load = load
    plot = plot

    def __init__(self, *args, **kwargs):
        self.model = type(self)._model_class(*args, **kwargs)
        self.model_name = type(self)._model_class.__name__
        self.optimiser = type(self)._optimiser
        self.scheduler = type(self)._scheduler
        self.criterion = type(self)._criterion
        self.data_dir = Path(str(type(self)._data_dir))
        self.verbose = type(self)._verbose
        self.history = {}

        # Initialise criterion, optimiser and scheduler
        if isinstance(self.criterion, str):
            self.criterion = str2crit(self.criterion)
        if isinstance(self.optimiser, str):
            self.optimiser = str2optim(self.optimiser, self.model)
        if isinstance(self.scheduler, str):
            self.scheduler = str2sched(self.scheduler, self.optimiser)

        # Set up logging
        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)
        self.logger = logging.getLogger()

        # Set logging level
        if self.verbose == 0:
            self.logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif self.verbose == 2:
            self.logger.setLevel(logging.DEBUG)

    def trainable_params(self) -> int:
        ''' Returns the number of trainable parameters in the model. '''
        return sum(param.numel() for param in self.model.parameters() 
                if param.requires_grad)

    def is_cuda(self) -> bool:
        return next(self.model.parameters()).is_cuda

    def forward(self, *args, **kwargs) -> Tensor:
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
