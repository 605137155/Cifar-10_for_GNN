import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def getCifar10():

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_train_data = cifar_train.data.reshape((50000, 1024, 3))


    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)
    cifar_test_data = cifar_test.data.reshape((10000, 1024, 3))


    return cifar_train_data, cifar_train.targets, cifar_test_data, cifar_test.targets


