import torch
import torchvision
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

def load_data():

    # Instantiate plain sets without normalization applied to get mean & std
    data_train = torchvision.datasets.MNIST(root = './data',
                                            train = True,
                                            download = True,
                                            transform = ToTensor())
    data_test = torchvision.datasets.MNIST(root = './data',
                                           train = False,
                                           download = True,
                                           transform = ToTensor())

    # Get mean & std for both sets
    data_train = next(iter(DataLoader(data_train, batch_size=len(data_train), num_workers=1)))[0]
    data_test  = next(iter(DataLoader(data_test,  batch_size=len(data_test),  num_workers=1)))[0]
    mean_train = data_train.mean()
    mean_test  = data_test .mean()
    std_train  = data_train.std()
    std_test   = data_test .std()

    # Return sets with normalizations applied
    trans_train = Compose([Resize((32,32)),
                           ToTensor(),
                           Normalize(mean=(mean_train,), std=(std_train,))])
    trans_test  = Compose([Resize((32,32)),
                           ToTensor(),
                           Normalize(mean=(mean_test, ), std=(std_test, ))])
    data_train = torchvision.datasets.MNIST(root = './data',
                                            train = True,
                                            download = False,
                                            transform = trans_train)
    data_test = torchvision.datasets.MNIST(root = './data',
                                           train = False,
                                           download = False,
                                           transform = trans_test)
    return data_train, data_test


def get_dataset(batch_size):
    data_train, data_test = load_data()
    train_batch_size = batch_size if batch_size > 0 else len(data_train)
    test_batch_size = batch_size if batch_size > 0 else len(data_test)
    train_loader = torch.utils.data.DataLoader(dataset = data_train,
                                               batch_size = train_batch_size,
                                               shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = data_test,
                                              batch_size = test_batch_size,
                                              shuffle = True)
    return train_loader, test_loader
