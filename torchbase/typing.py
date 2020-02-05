from typing import Callable, List, Union, Iterable, TypeVar, Sequence
from pathlib import Path
from torch import Tensor

# Basic types
nInt = Union[int, None]
nFloat = Union[float, None]
nBool = Union[bool, None]
nStr = Union[str, None]
Numeric = Union[int, float]
nNumeric = Union[Numeric, None]

# Lists of basic types
Ints = List[int]
Floats = List[float]
Bools = List[bool]
Strs = List[str]
Numerics = List[Numeric]

Tensor = Sequence[Numeric]
Wrapper = TypeVar('Wrapper')
Module = TypeVar('Module')
Decorator = Callable[[Callable], Callable]
Pathlike = Union[Path, str]

Metric = Callable[[Tensor, Tensor], float]
Metriclike = Union[Metric, str]
Metrics = List[Metric]
Metriclikes = List[Metriclike]

Optimiser = TypeVar('Optimiser')
Optimiserlike = Union[Optimiser, str]
nOptimiserlike = Union[Optimiser, str, None]

Scheduler = TypeVar('Scheduler')
Schedulerlike = Union[Scheduler, str]
nSchedulerlike = Union[Scheduler, str, None]

DataLoader = Iterable[Tensor]
nDataLoader = Union[DataLoader, None]
