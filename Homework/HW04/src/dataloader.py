import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def load_cifar10_subset(path, subset_classes=10, train_percent=0.1, seed=42):
	"""
	加载CIFAR-10子集数据（不含验证集）
	参数:
		path: 数据集路径
		subset_classes: 使用的类别数量（前 n 类）
		train_percent: 从训练集中采样的比例
		seed: 随机种子
	返回:
		train_dataset: 训练集子集
		test_dataset: 测试集（完整测试集）
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)

	transform = transforms.Compose([
		transforms.ToTensor()
	])

	# 加载 CIFAR-10 数据集
	train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

	# 筛选前 subset_classes 类的样本
	train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < subset_classes]
	test_indices = [i for i, (_, label) in enumerate(test_dataset) if label < subset_classes]

	# 从训练集中采样
	num_samples = int(train_percent * len(train_indices))
	sampled_indices = np.random.choice(train_indices, num_samples, replace=False)
	train_subset = Subset(train_dataset, sampled_indices)

	# 创建测试集子集
	test_subset = Subset(test_dataset, test_indices)

	return train_subset, test_subset

# 数据增强定义
def get_augmentations(name="basic", normalize=True):
	"""
	定义SimCLR数据增强
	参数:
		normalize: 是否添加Normalize
	返回:
		augmentation: 数据增强操作
	"""
	transform_list = []

	if name in ["basic", "color", "gray", "strong"]:
		transform_list.extend([
			transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),	# 1. 随机调整大小并裁剪到32x32
			transforms.RandomHorizontalFlip(p=0.5)])					# 2. 以0.5的概率水平翻转图像

	if name in ["color", "gray", "strong"]:
		transform_list.append(
			transforms.RandomApply(										# 3. 以0.8的概率应用颜色抖动
				[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))

	if name in ["gray", "strong"]:
		transform_list.append(transforms.RandomGrayscale(p=0.2))		# 4. 以0.2的概率转换为灰度图

	if name == 'strong':
		transform_list.extend([
			transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
			transforms.RandomSolarize(threshold=0.5)])

	transform_list.append(transforms.ToTensor())

	if normalize:
		transform_list.append(transforms.Normalize(						#cifar-10数据集的均值和标准差
			(0.4914, 0.4822, 0.4465),
			(0.2023, 0.1994, 0.2010)))

	return transforms.Compose(transform_list)
