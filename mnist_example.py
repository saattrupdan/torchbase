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

@magic('categorical_cross_entropy', metrics = 'accuracy_with_logits')
class Net(torch.nn.Module):
    def __init__(self, dim: int = 100):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, dim)
        self.out = torch.nn.Linear(dim, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = torch.nn.functional.gelu(self.fc(x))
        return self.out(x)

if __name__ == '__main__':
    train_dl, val_dl = get_mnist_dataloaders()
    net = Net()
    net.fit(train_dl, val_dl)
    net.plot(['accuracy', 'val_accuracy'])
