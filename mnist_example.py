from torchbase import magic
import torch

def get_mnist_dataloaders():
    ''' Fetch a preprocessed MNIST training- and validation dataloader. '''
    import torchvision

    train = torchvision.datasets.MNIST('.data', train = True,
        download = True, transform = torchvision.transforms.ToTensor())
    val = torchvision.datasets.MNIST('.data', train = False,
        download = True, transform = torchvision.transforms.ToTensor())

    train_dl = torch.utils.data.DataLoader(train, batch_size = 32, 
        shuffle = True)
    val_dl = torch.utils.data.DataLoader(val, batch_size = 32, 
        shuffle = True)

    return train_dl, val_dl

@magic(criterion = 'categorical_cross_entropy')
class MLP(torch.nn.Module):
    def __init__(self, dim: int = 100) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, dim)
        self.out = torch.nn.Linear(dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.gelu(self.fc(x))
        return torch.nn.functional.log_softmax(self.out(x), dim = -1)

if __name__ == '__main__':
    train_dl, val_dl = get_mnist_dataloaders()
    model = MLP(dim = 100).fit(train_dl, val_dl)
    model.plot(['loss'])
