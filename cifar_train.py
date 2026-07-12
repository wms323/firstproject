import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#数据准备
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#下载CIFAR10数据集
train_dataset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

#DataLoader
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=False)

