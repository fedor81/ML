import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


ROOT = "./basics-deep-learning-and-AI/data/"


class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFARDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_loaders(train_dataset, test_dataset, batch_size=64, num_workers=0):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    return get_loaders(train_dataset, test_dataset, batch_size=batch_size)


def get_cifar_loaders(batch_size=64):
    """The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = CIFARDataset(train=True, transform=transform)
    test_dataset = CIFARDataset(train=False, transform=transform)

    return get_loaders(train_dataset, test_dataset, batch_size=batch_size)
