import torch
from .decorators import changename
from .typing import *

class MetricObject:
    def __init__(self, metric: Functionlike, wrapper: Wrapper, val: bool):
        from .utils import str2function
        self.metric = str2function(metric, wrapper)
        
        try:
            self.name = self.metric.__name__.lower()
        except AttributeError:
            self.name = type(self.metric).__name__.lower()

        if val: self.name = 'val_' + self.name

    def __call__(self, pred: Tensor, true: Tensor) -> float:
        return self.metric(pred, true)

def accuracy(pred: Tensor, true: Tensor) -> float:
    if pred.shape != true.shape:
        true = torch.nn.functional.one_hot(true, num_classes = pred.shape[-1])
    pred = torch.gt(pred, 0.5)
    correct = torch.eq(torch.sum(torch.eq(pred, ~true.bool()), dim = 1), 0)
    return torch.mean(correct.float()).item()

@changename('accuracy')
def accuracy_with_logits(pred: Tensor, true: Tensor) -> float:
    return accuracy(torch.nn.functional.softmax(pred, dim = -1), true)

def samples_f1(pred: Tensor, true: Tensor) -> float:
    pred, true = torch.gt(pred, 0.5), torch.gt(true, 0.5)
    tp = torch.sum(pred & true, dim = 1).float()
    false_preds = torch.sum(torch.eq(pred, ~true), dim = 1).float()
    denom = 2 * tp + false_preds
    denom = torch.where(denom == 0, torch.ones(denom.size()), denom)
    return torch.mean(2 * tp / denom).item()

@changename('samples_f1')
def samples_f1_with_logits(pred: Tensor, true: Tensor) -> float:
    return samples_f1(torch.sigmoid(pred), true)
