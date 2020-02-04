from torchbase import magic
import torch

@magic(criterion = 'binary_cross_entropy', optimiser = 'adamw')
class MLP(torch.nn.Module):
    def __init__(self, dim: int = 100) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, dim)
        self.out = torch.nn.Linear(dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.gelu(self.fc(x))
        return torch.sigmoid(self.out(x))

if __name__ == '__main__':
    import torchvision

    def one_hot(n: int, nclasses: int = 10):
        tensor = torch.zeros(nclasses)
        tensor[n] = 1.
        return tensor

    train = torchvision.datasets.MNIST('.data', train = True,
        download = True, transform = torchvision.transforms.ToTensor(),
        target_transform = one_hot)
    val = torchvision.datasets.MNIST('.data', train = False,
        download = True, transform = torchvision.transforms.ToTensor(),
        target_transform = one_hot)

    train_dl = torch.utils.data.DataLoader(train, batch_size = 32, 
        shuffle = True)
    val_dl = torch.utils.data.DataLoader(val, batch_size = 32, 
        shuffle = True)

    model = MLP(dim = 100)
    print(model)
    model.fit(train_dl, val_dl, patience = 1)
    model.plot(['loss'])
