from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary


def get_and_transform_the_data():
	use_cuda = torch.cuda.is_available()

	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)
	
	SEED=1
	
	torch.manual_seed(SEED)

	if cuda:
		torch.cuda.manual_seed(SEED)

	transform = transforms.Compose([transforms.RandomRotation((-7.0, 7.0)),transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)

	trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
	testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
                                       

	train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	test_loader= torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	return trainset, testset, train_loader, test_loader, classes
    