import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 定义前馈神经网络
class FNN(nn.Module):
	def __init__(self, input_dim, hidden_layers, activation_fn):
		super(FNN, self).__init__()
		layers = []
		prev_dim = input_dim

		# 添加隐藏层
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(prev_dim, hidden_dim))
			layers.append(activation_fn())
			prev_dim = hidden_dim

		# 添加输出层
		layers.append(nn.Linear(prev_dim, 1))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


# 加载数据集
def load_dataset():
	boston = fetch_openml(name="boston", version=1, as_frame=True)
	print(f"特征字段: {boston.feature_names}")
	key = input("请输入特征字段: ")

	X, y, loaded = [], [], key in boston.feature_names

	if loaded:
		df = boston.frame
		X = df[boston.feature_names].values
		y = df[key].values.reshape(-1, 1)

		# 数据标准化
		scaler_X = StandardScaler()
		scaler_y = StandardScaler()
		X = scaler_X.fit_transform(X)
		y = scaler_y.fit_transform(y)
	else:
		print(f"{key} 不在 boston.feature_names 中")

	return X, y, loaded


# 数据集划分
def divide_dataset(X, y, test_size):
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=test_size, random_state=42)

	# 转换为 PyTorch 张量
	X_train, y_train, X_val, y_val, X_test, y_test = map(
		lambda x: torch.tensor(x, dtype=torch.float32),
		(X_train, y_train, X_val, y_val, X_test, y_test)
	)

	return X_train, X_val, X_test, y_train, y_val, y_test


# 选择训练模式

def operation():
	return int(input("请输入 0: 比较训练参数; 1: 最终训练: "))


# 训练函数
def train_model(
		model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, batch_size):
	train_losses, val_losses = [], []
	dataset = torch.utils.data.TensorDataset(X_train, y_train)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	for epoch in range(epochs):
		model.train()
		train_loss = 0

		for batch_X, batch_y in dataloader:
			optimizer.zero_grad()
			predictions = model(batch_X)
			loss = criterion(predictions, batch_y)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		train_losses.append(train_loss / len(dataloader))

		# 计算验证集损失
		model.eval()
		with torch.no_grad():
			val_predictions = model(X_val)
			val_loss = criterion(val_predictions, y_val).item()
			val_losses.append(val_loss)

		print(
			f"Epoch {epoch + 1 :03d}/{epochs},"
			f"Train Loss: {train_losses[-1]:.4f},"
			f"Val Loss: {val_losses[-1]:.4f}")

	return train_losses, val_losses


# 训练不同参数的模型
def run_experiments(X_train, y_train, X_val, y_val, epochs, batch_size, parameters):
	experiment_results_depth, experiment_results_lr, experiment_results_act = {}, {}, {}
	dim = X_train.shape[1]

	depths = parameters["depths"]
	activation_functions = parameters["activation_functions"]
	learning_rates = parameters["learning_rates"]
	criterion = nn.MSELoss()

	# 深度对比实验
	for depth in depths:
		print(f"\nTraining FNN with layers: {depth}")
		model = FNN(input_dim=dim, hidden_layers=depth, activation_fn=nn.ReLU)
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		train_losses, val_losses = train_model(
			model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, batch_size)
		experiment_results_depth[f"Depth {depth}"] = (train_losses, val_losses)
	plot_results(experiment_results_depth, "Comparison of Different Depths")

	# 激活函数对比实验
	for act_fn in activation_functions:
		print(f"\nTraining FNN with activation function: {act_fn.__name__}")
		model = FNN(input_dim=dim, hidden_layers=[64, 32], activation_fn=act_fn)
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		train_losses, val_losses = train_model(
			model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, batch_size)
		experiment_results_act[f"Activation {act_fn.__name__}"] = (train_losses, val_losses)
	plot_results(experiment_results_act, "Comparison of Different Activation Functions")

	# 学习率对比实验
	for lr in learning_rates:
		print(f"\nTraining FNN with learning rate: {lr}")
		model = FNN(input_dim=dim, hidden_layers=[64, 32], activation_fn=nn.ReLU)
		optimizer = optim.Adam(model.parameters(), lr=lr)
		train_losses, val_losses = train_model(
			model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, batch_size)
		experiment_results_lr[f"LR {lr}"] = (train_losses, val_losses)
	plot_results(experiment_results_lr, "Comparison of Different Learning Rates")


# 画图对比
def plot_results(experiment_results, title):
	plt.figure(figsize=(12, 6))
	for key, (train_losses, val_losses) in experiment_results.items():
		plt.plot(val_losses, label=key)
	plt.xlabel("Epochs")
	plt.ylabel("Validation Loss")
	plt.title(title)
	plt.legend()
	plt.show()


# 评估最终模型
def evaluate_model(
		hidden_layers, activation_fn, lr,
		X, X_test, y_test, X_train, y_train, X_val, y_val,
		epochs,  batch_size):
	evaluate_results = {}
	final_model = FNN(X.shape[1], hidden_layers, activation_fn)
	final_optimizer = optim.Adam(final_model.parameters(), lr)
	final_criterion = nn.MSELoss()
	train_losses, val_losses = train_model(
		final_model, final_optimizer, final_criterion,
		X_train, y_train, X_val, y_val, epochs, batch_size)
	evaluate_results["evaluate"] = (train_losses, val_losses)
	plot_results(evaluate_results, "Selected Parameters")

	final_model.eval()
	with torch.no_grad():
		predictions = final_model(X_test)
		test_loss = final_criterion(predictions, y_test).item()
	print(f"\nFinal Test Loss: {test_loss:.4f}")
	return test_loss


# 主函数
def main():
	test_size = 0.2
	epochs, batch_size = 100, 32

	# 设置实验参数
	parameters  = {
		"depths": [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32], [512, 256, 128, 64, 32]],
		"activation_functions": [nn.ReLU, nn.Sigmoid, nn.Tanh],
		"learning_rates": [0.01, 0.001, 0.0001]}

	# 设置最终参数
	hidden_layers = [64, 32]
	active_fn = nn.ReLU
	learning_rates = 0.001

	X, y, loaded = load_dataset()

	if loaded:
		X_train, X_val, X_test, y_train, y_val, y_test = divide_dataset(X, y, test_size)

		if operation() == 0:
			run_experiments(X_train, y_train, X_val, y_val, epochs, batch_size, parameters)

		else:
			evaluate_model(
				hidden_layers, active_fn, learning_rates,
				X, X_test, y_test, X_train, y_train, X_val, y_val, epochs, batch_size)


if __name__ == '__main__':
	main()
