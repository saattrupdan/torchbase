from typing import Callable, Sequence, Union, Iterable, TypeVar, Sequence 
from pathlib import Path
import torch
import torch.utils.data as data
import torch.optim.optimizer as optim
import torch.optim.lr_scheduler as sched
import torch.nn.modules.loss as loss

# Basic types
nInt = Union[int, None]
nFloat = Union[float, None]
nBool = Union[bool, None]
nStr = Union[str, None]
Numeric = Union[int, float]
nNumeric = Union[Numeric, None]

# Sequences of basic types
Ints = Sequence[int]
Floats = Sequence[float]
Bools = Sequence[bool]
Strs = Sequence[str]
Numerics = Sequence[Numeric]
Iterable = Iterable

# Tensors
Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# Functions
Function = Callable
Functionlike = Union[Function, str]

# Criterions
Criterion = loss._Loss
Criterionlike = Union[loss._Loss, str]

# A type with the same signature as Metric, to fool MyPy
class Metric:
    __name__: str = ''
    def __init__(self, *args, **kwargs):
        self.metric = None
        self.name = None
    def __call__(self, pred: Tensor, true: Tensor) -> float: pass

# Metrics
Metriclike = Union[Metric, str]
Metrics = Sequence[Metric]
Metriclikes = Sequence[Metriclike]
MetriclikesOrString = Union[Metriclikes, str]
nMetriclikesOrString = Union[Metriclikes, str, None]

# Optimisers
Optimiser = optim.Optimizer
Optimiserlike = Union[optim.Optimizer, str]
nOptimiserlike = Union[optim.Optimizer, str, None]

# Schedulers
Scheduler = Union[sched._LRScheduler, sched.ReduceLROnPlateau]
Schedulerlike = Union[sched._LRScheduler, str]
nSchedulerlike = Union[sched._LRScheduler, str, None]

# DataLoaders
DataLoader = data.DataLoader
nDataLoader = Union[data.DataLoader, None]

# Misc
Module = torch.nn.Module
Decorator = Callable
Pathlike = Union[Path, str]

# A type with the same signature as ModuleWrapper, to fool MyPy
class Wrapper:
    def __init__(self, *args, **kwargs):
        self.model = None
        self.criterion = None
        self.optimiser = None
        self.scheduler = None
        self.metrics = None
        self.history = None
        self.data_dir = None
        self.model_name = None
        self.logger = None
    def trainable_params(self) -> int: pass
    def fit(*args, **kwargs): pass
    def save(self, postfix: str): pass
    def save_model(self, postfix: str): pass
    def load(self, *args, **kwargs): pass
    def plot(self, *args, **kwargs): pass
    def is_cuda(self) -> bool: pass
    def forward(self, tensor: Tensor) -> Tensor: pass
    def __call__(self, tensor: Tensor) -> Tensor: pass
    def __repr__(*args, **kwargs): pass
    def train(*args, **kwargs): pass
    def eval(*args, **kwargs): pass
    def to(*args, **kwargs): pass
    def cuda(*args, **kwargs): pass
    def cpu(*args, **kwargs): pass
    def load_state_dict(*args, **kwargs): pass
