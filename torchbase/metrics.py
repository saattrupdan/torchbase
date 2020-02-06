import torch
from .typing import *

def accuracy(pred: Tensor, true: Tensor) -> float:
    correct = torch.eq(torch.sum(torch.eq(pred, ~true.bool()), dim = 1), 0)
    return torch.mean(correct.float()).item()

def samples_f1(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.bool(), true.bool()
    tp = torch.sum(pred & true, dim = 1).float()
    false_preds = torch.sum(torch.eq(pred, ~true), dim = 1).float()
    denom = 2 * tp + false_preds
    denom = torch.where(denom == 0, torch.ones(denom.size()), denom)
    return torch.mean(2 * tp / denom).item()
