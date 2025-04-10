import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from load_dataset import train_loader, test_loader
from cnn_builder import build_cnn


# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = build_cnn(
	1, [16, 32], 3,
	True, False, 'max', 10
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练与验证
best_acc = 0.0
best_model = copy.deepcopy(model.state_dict())

for epoch in range(30):
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

# 加载最佳模型 & 测试
model.load_state_dict(best_model)
model.eval()
errors = []

test_correct = 0
with torch.no_grad():
	for x, y in test_loader:
		x, y = x.to(device), y.to(device)
		output = model(x)
		test_correct += (output.argmax(1) == y).sum().item()
		predicts = output.argmax(1)

		for i in range(len(y)):
			if predicts[i] != y[i]:
				errors.append((x[i].cpu(), y[i].item(), predicts[i].item()))

test_acc = test_correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")

folder_path = 'dataset/errors'
if os.path.exists(folder_path):
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		if os.path.isfile(file_path):
			os.remove(file_path)
else:
	os.makedirs(folder_path)

for i in range(len(errors)):
	img, label, predict = errors[i]
	file_name = f"error_{i:02d}_true{label}_predict{predict}.png"
	plt.imshow(img.squeeze(), cmap='gray')
	plt.axis('off')
	plt.savefig(os.path.join(folder_path, file_name))
	plt.close()
