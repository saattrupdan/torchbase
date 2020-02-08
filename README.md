<h1 align=center>
  <img alt="TorchBase" src="gfx/title.png" height=60px>
</h1>

A simple decorator that allows you to `fit` and `plot` your PyTorch model.

Requires `torch`, `tqdm` and `matplotlib`.

**This project is in early development. Brace yourselves, breaking changes are coming.**

## Basic MNIST example
See `mnist_example.py` for the full example, but here are the essential parts:

```python
>>> from torchbase import magic
>>> 
>>> @magic('categorical_cross_entropy', metrics = 'accuracy_with_logits')
... class Net(torch.nn.Module):
...   def __init__(self):
...     super().__init__()
...     self.fc = torch.nn.Linear(28 * 28, 100)
...     self.out = torch.nn.Linear(100, 10)
... 
...   def forward(self, x):
...     x = torch.flatten(x, start_dim = 1)
...     x = torch.nn.functional.gelu(self.fc(x))
...     return self.out(x)
... 
>>> # Get preprocessed MNIST data
>>> train_dl, val_dl = get_mnist_dataloaders()
>>> 
>>> net = Net()
>>> net.fit(train_dl, val_dl)
[INFO] Training on 60,000 samples and validation on 10,016 samples
[INFO] Number of trainable parameters: 79,510
Epoch  0 - loss 0.1930 - accuracy 0.9342 - val_loss 0.1638 - val_accuracy 0.9427: 100%|███████████████████████████████| 60000/60000 [00:08<00:00, 6695.25it/s]
(...)
Epoch 20 - loss 0.0090 - accuracy 0.9967 - val_loss 0.0879 - val_accuracy 0.9808: 100%|███████████████████████████████| 60000/60000 [00:08<00:00, 6921.11it/s]
>>> net.plot(['accuracy', 'val_accuracy'])
```
![Plot showing the accuracy and validation accuracy by epoch](gfx/mnist.png)

## Documentation

### `magic` decorator parameters
- criterion: `str` or valid loss function (default = `binary_cross_entropy`)
- optimiser: `str` or PyTorch `Optimizer` (default = `adamw`)
- scheduler: `str` or PyTorch scheduler (default = `reduce_on_plateau`)
- metrics: `str` or function, or an iterable of such (default = `[]`)
- learning_rate: `float` (default = `3e-4`)
- monitor: `str` or `None` (default = `None`)
- minimise_monitor: `bool` or `None` (default = `None`)
- target_value: `float` or `None` (default = `None`)
- patience: `int` or `None` (default = `9`)
- tensorboard: `bool` (default = `False`)
- smoothing: `float` (default = `0.99`)
- save_model: `bool` (default = `True`)
- overwrite: `bool` (default = `True`)
- data_dir: `str` or pathlib.Path (default = '.data')
- verbose: `int` (default = `1`)

### `fit` method parameters
- train_loader: PyTorch `DataLoader`
- val_loader: PyTorch `DataLoader` or `None` (default = `None`)
- epochs: `int` or `None` (default = `None`)

### `plot` method parameters
- metrics: `str`, iterable of `str`s or `None` (default = `None`)
- save_to: `str` or `None` (default = `None`)
- title: `str` or `None` (default = `Model performance by epoch`)
- xlabel: `str` or `None` (default = `epochs`)
- ylabel: `str` or `None` (default = `score`)
- show_legend: `bool` (default = `True`)
- show_plot: `bool` (default = `True`)
- plot_style: `str` (default = `ggplot`)
