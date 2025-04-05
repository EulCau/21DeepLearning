import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from load_dataset import train_loader, test_loader
import copy

# 1. 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输出: 16×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 输出: 16×14×14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 输出: 32×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 输出: 32×7×7

            nn.Flatten(),                               # 输出: 32×7×7 = 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# 2. 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 训练与验证
best_acc = 0.0
best_model = copy.deepcopy(model.state_dict())

for epoch in range(10):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_correct += (output.argmax(1) == y).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item() * x.size(0)
            val_correct += (output.argmax(1) == y).sum().item()

    val_loss /= len(test_loader.dataset)
    val_acc = val_correct / len(test_loader.dataset)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(model.state_dict())

# 4. 加载最佳模型 & 测试
model.load_state_dict(best_model)
model.eval()

test_correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        test_correct += (pred.argmax(1) == y).sum().item()

test_acc = test_correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")
