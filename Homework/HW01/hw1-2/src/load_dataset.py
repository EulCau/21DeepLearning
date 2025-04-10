import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_subset(dataset, ratio):
	if ratio>=1.0:
		return dataset

	indices = np.random.choice(
		len(dataset),
		size=int(len(dataset)*ratio),
		replace=False
	)

	return Subset(dataset, indices)


def load_data(seed=42, ratio=0.1, data_path='../dataset'):
	torch.manual_seed(seed)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	train_dataset = datasets.MNIST(
		root=data_path,
		train=True,
		download=True,
		transform=transform
	)
	test_dataset = datasets.MNIST(
		root=data_path,
		train=False,
		transform=transform
	)

	train_subset = get_subset(train_dataset, ratio)
	test_subset = get_subset(test_dataset, ratio)

	train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
	test_loader = DataLoader(test_subset, batch_size=16)

	print(f"训练样本数: {len(train_subset)}, 测试样本数: {len(test_subset)}")
	return train_loader, test_loader
