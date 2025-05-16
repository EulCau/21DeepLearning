import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from dataloader import load_cifar10_subset, get_augmentations


# 创建SimCLR数据增强的数据集
class SimCLRDataset(Dataset):
	"""生成SimCLR双视图增强数据的Dataset类"""

	def __init__(self, dataset):
		self.dataset = dataset
		self.augment = get_augmentations(normalize=True)

	def __getitem__(self, index):
		img, _ = self.dataset[index]  # 忽略原始标签
		img_pil = transforms.ToPILImage()(img)
		return (self.augment(img_pil), self.augment(img_pil)), 0  # 返回两个增强视图

	def __len__(self):
		return len(self.dataset)


class SimCLR(nn.Module):
	"""SimCLR模型：基础编码器+投影头"""

	def __init__(self, base_encoder=models.resnet18, projection_dim=128):
		super().__init__()
		# 加载基础编码器
		self.encoder = base_encoder(weights=None)
		self.encoder.conv1 = nn.Conv2d(
			3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 适配 CIFAR-10
		self.encoder.maxpool = nn.Identity()
		self.encoder.fc = nn.Identity()  # 移除最后的全连接层

		# 投影头（带BN的非线性MLP）
		self.projection = nn.Sequential(
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, projection_dim))

	def forward(self, x):
		h = self.encoder(x)
		return self.projection(h)


class LinearClassifier(nn.Module):
	"""线性分类器"""

	def __init__(self, encoder, num_classes=10):
		super().__init__()
		self.encoder = encoder
		for param in self.encoder.parameters():  # 冻结编码器参数
			param.requires_grad = False
		self.fc = nn.Linear(512, num_classes)  # 接线性分类层

	def forward(self, x):
		with torch.no_grad():
			h = self.encoder(x)
		return self.fc(h)


def nt_xent_loss(z, temperature=0.5, device="cuda"):
	"""NT Cross Entropy 对比损失"""
	batch_size = z.shape[0] // 2  # 原始批次大小 N
	z = nn.functional.normalize(z, dim=1)

	# 计算相似度矩阵 [2N x 2N]
	sim_matrix = torch.mm(z, z.T) / temperature

	# 生成正样本对标签（每个样本的正对索引）
	labels = torch.arange(2 * batch_size, device=device)
	labels = (labels + batch_size) % (2 * batch_size)  # 正对索引偏移

	# 计算交叉熵损失
	loss = nn.CrossEntropyLoss()(sim_matrix, labels)
	return loss

def train_simclr(model, train_loader, optimizer, epoch, device, config):
	"""SimCLR训练"""
	model.train()
	total_loss = 0.0

	for (view1, view2), _ in train_loader:  # 忽略原始标签
		x = torch.cat([view1, view2], dim=0).to(device)  # 合并两个视图
		optimizer.zero_grad()

		z = model(x)  # 前向传播
		loss = nt_xent_loss(z, config["temperature"], device)

		loss.backward()
		optimizer.step()

		total_loss += loss.item() * x.size(0)

	avg_loss = total_loss / len(train_loader.dataset)
	print(f"Epoch [{epoch + 1}/{config['epochs']}] Loss: {avg_loss:.4f}")

	return avg_loss


def evaluate_linear(model, test_loader, device):
	"""线性评估分类器"""
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += torch.Tensor(predicted == labels).sum().item()

	accuracy = 100 * correct / total
	print(f"Test Accuracy: {accuracy:.2f}%")
	return accuracy


def show_plot(data, label, title, x_label, y_label, fig_name):
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(data) + 1), data, 'b-', linewidth=2, label=label)
	plt.title(title, fontsize=14)
	plt.xlabel(x_label, fontsize=12)
	plt.ylabel(y_label, fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend()
	plt.savefig(fig_name, dpi=300, bbox_inches='tight')
	plt.show()


def main():
	# 设备配置
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 配置参数
	config = {
		"batch_size": 256,
		"projection_dim": 128,
		"temperature": 0.1,
		"lr": 3e-2,
		"epochs": 100,
		"eval_epochs": 50,
		"num_workers": 4
	}

	# 加载数据
	train_data, test_data = load_cifar10_subset(path= '../dataset', subset_classes=10, train_percent=1)

	simclr_dataset = SimCLRDataset(train_data)
	train_loader = DataLoader(simclr_dataset, batch_size=config["batch_size"],
							  shuffle=True, num_workers=config["num_workers"], drop_last=True)

	# 初始化模型和优化器
	model = SimCLR(projection_dim=config["projection_dim"]).to(device)
	optimizer = optim.Adam(model.parameters(), lr=config["lr"])

	# 预训练阶段
	print("=== Starting SimCLR Pretraining ===")
	losses = []
	for epoch in range(config["epochs"]):
		losses.append(train_simclr(model, train_loader, optimizer, epoch, device, config))
	show_plot(
		losses, "Training Loss", "SimCLR Pretraining Loss Curve",
		"Epoch", "Loss", "../result/pretrain_loss.png")

	# 线性评估阶段
	print("\n=== Starting Linear Evaluation ===")
	test_loader = DataLoader(
		test_data, batch_size=config["batch_size"],
		num_workers=config["num_workers"])
	classifier = LinearClassifier(model.encoder).to(device)
	optimizer_cls = optim.Adam(classifier.parameters(), lr=3e-4)
	criterion = nn.CrossEntropyLoss()

	# 训练分类器
	accuracies = []
	for epoch in range(config["eval_epochs"]):
		classifier.train()
		for images, labels in DataLoader(train_data, batch_size=config["batch_size"], shuffle=True):
			images, labels = images.to(device), labels.to(device)
			optimizer_cls.zero_grad()
			outputs = classifier(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer_cls.step()

		# 评估
		accuracies.append(evaluate_linear(classifier, test_loader, device))
	show_plot(
		accuracies, "Test Accuracy", "Linear Evaluation Accuracy Curve",
		"Epoch", "Accuracy (%)", "../result/linear_eval_accuracy.png")


if __name__ == "__main__":
	main()
