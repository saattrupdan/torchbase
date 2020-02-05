# TorchBase

A PyTorch wrapper to spice up your models.

A simple decorator on your model allows you to call `fit` and `plot`.

**This project is in early development, many breaking changes will happen and many bugs probably already happened**

## MNIST example
See `mnist_example.py` for the full example, but here are the essential parts:

```python
from torchbase import magic

@magic(criterion = 'categorical_cross_entropy')
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(28 * 28, 100)
    self.out = torch.nn.Linear(100, 10)

  def forward(self, x):
    x = torch.flatten(x, start_dim = 1)
    x = torch.nn.functional.gelu(self.fc(x))
    return self.out(x)

# Get preprocessed MNIST data
train_dl, val_dl = get_mnist_dataloaders()

net = Net()
net.fit(train_dl, val_dl)
net.plot('val_accuracy')
```
