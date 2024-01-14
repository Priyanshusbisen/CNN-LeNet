import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

def main():
    train_data = datasets.FashionMNIST(
       root = '/Users/priyanshusingh/Desktop/PytorchPrac/torchProject/Dataset',
       train=True,
       download=True,
       transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root= '/Users/priyanshusingh/Desktop/PytorchPrac/torchProject/Dataset',
        train=False,
        download=True,
        transform=ToTensor
        
    )


if __name__ == '__main__':
    main()
