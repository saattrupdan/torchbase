from typing import Callable, Sequence, Union, Iterable, TypeVar, Sequence
from pathlib import Path
import torch

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

# Tensors
Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# Metrics
Metric = Callable[[Numerics, Numerics], float]
Metriclike = Union[Metric, str]
Metrics = Sequence[Metric]
Metriclikes = Sequence[Metriclike]

# Optimisers
Optimiser = TypeVar('Optimiser')
Optimiserlike = Union[Optimiser, str]
nOptimiserlike = Union[Optimiser, str, None]

# Schedulers
Scheduler = TypeVar('Scheduler')
Schedulerlike = Union[Scheduler, str]
nSchedulerlike = Union[Scheduler, str, None]

# DataLoaders
DataLoader = Iterable[Tensor]
nDataLoader = Union[DataLoader, None]

# Misc
Module = torch.nn.Module
Wrapper = TypeVar('Wrapper')
Decorator = Callable[[Callable], Callable]
Pathlike = Union[Path, str]
