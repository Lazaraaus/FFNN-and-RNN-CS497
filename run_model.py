import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchtext.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from data import my_dataset

def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None,
              batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):

    """
        model: 
        running_mode:
        train_set, valid_set, test_set:
        batch_size:
        learning_rate:
        n_epochs:
        stop_thr:
        shuffle:

    """
    pass


def _train(model, data_loader, optimizer, device=torch.device('cuda')):
    pass

def _test(model, data_loader, optimizer, device=torch.device('cuda')):
    pass


print(torch.cuda.device(0))
#print(os.getcwd())
#DataSet = my_dataset.MyDataset('train')
#train_dataloader = DataLoader(DataSet)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.cuda.device(0)  
#for batch_idx, (img, label) in enumerate(train_dataloader):

a = torch.rand(10)
a = a.to(device1)
b = torch.tensor([1])
b = b.to(device1)
    