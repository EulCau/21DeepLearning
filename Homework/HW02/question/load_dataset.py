import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 设置随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载完整数据集
train_dataset = datasets.MNIST(
    root='./dataset',
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./dataset',
    train=False, 
    transform=transform
)

# 随机抽取1%数据
def get_subset(dataset, ratio=0.01):
    indices = np.random.choice(
        len(dataset), 
        size=int(len(dataset)*ratio), 
        replace=False
    )
    return Subset(dataset, indices)

train_subset = get_subset(train_dataset)
test_subset = get_subset(test_dataset)

# 创建DataLoader
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=16)

print(f"训练样本数: {len(train_subset)}, 测试样本数: {len(test_subset)}")