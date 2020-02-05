from typing import Callable, Sequence, Union, Iterable, TypeVar, Sequence
from pathlib import Path
from torch import Tensor

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

Tensor = Sequence[Numeric]
Wrapper = TypeVar('Wrapper')
Module = TypeVar('Module')
Decorator = Callable[[Callable], Callable]
Pathlike = Union[Path, str]

Metric = Callable[[Tensor, Tensor], float]
Metriclike = Union[Metric, str]
Metrics = Sequence[Metric]
Metriclikes = Sequence[Metriclike]

Optimiser = TypeVar('Optimiser')
Optimiserlike = Union[Optimiser, str]
nOptimiserlike = Union[Optimiser, str, None]

Scheduler = TypeVar('Scheduler')
Schedulerlike = Union[Scheduler, str]
nSchedulerlike = Union[Scheduler, str, None]

DataLoader = Iterable[Tensor]
nDataLoader = Union[DataLoader, None]
