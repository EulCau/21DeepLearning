import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_builder import build_cnn
from load_dataset import load_data


def train(train_loader, val_loader, model, criterion, optimizer):
	"""
	Train the CNN model using the given training and validation data loaders.

	Args:
		train_loader (DataLoader): DataLoader for training data.
		val_loader (DataLoader): DataLoader for validation data.
		model (nn.Module): The CNN model to train.
		criterion: Loss function.
		optimizer: Optimizer for training.

	Returns:
		dict: The state dictionary of the best-performing model on validation set.
	"""
	best_acc = 0.0
	best_model = copy.deepcopy(model.state_dict())
	device = next(model.parameters()).device

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
			train_correct += torch.eq(output.argmax(1), y).sum().item()

		train_loss /= len(train_loader.dataset)
		train_acc = train_correct / len(train_loader.dataset)

		model.eval()
		val_loss = 0.0
		val_correct = 0

		with torch.no_grad():
			for x, y in val_loader:
				x, y = x.to(device), y.to(device)
				output = model(x)
				loss = criterion(output, y)
				val_loss += loss.item() * x.size(0)
				val_correct += torch.eq(output.argmax(1), y).sum().item()

		val_loss /= len(val_loader.dataset)
		val_acc = val_correct / len(val_loader.dataset)

		print(
			f"Epoch {(epoch + 1):02d}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
			f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

		if val_acc > best_acc:
			best_acc = val_acc
			best_model = copy.deepcopy(model.state_dict())

	return best_model


def test(test_loader, model, best_model, data_path):
	"""
	Evaluate the best model on the test dataset and save misclassified images.

	Args:
		test_loader (DataLoader): DataLoader for test data.
		model (nn.Module): The CNN model.
		best_model (dict): The best model state dictionary from training.
		data_path (str): Path to the dataset for saving error images.
	"""
	model.load_state_dict(best_model)
	model.eval()
	errors = []
	device = next(model.parameters()).device

	test_correct = 0

	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device), y.to(device)
			output = model(x)
			test_correct += torch.eq(output.argmax(1), y).sum().item()
			predicts = output.argmax(1)

			for i in range(len(y)):
				if predicts[i] != y[i]:
					errors.append((x[i].cpu(), y[i].item(), predicts[i].item()))

	test_acc = test_correct / len(test_loader.dataset)
	print(f"Test Accuracy: {test_acc:.4f}")

	save_err_data(data_path, errors)


def save_err_data(data_path, errors):
	"""
	Save images that were misclassified by the model.

	Args:
		data_path (str): Root path to the dataset.
		errors (list): List of tuples (image tensor, true label, predicted label).
	"""
	err_path = os.path.join(data_path, 'errors')

	if os.path.exists(err_path):
		for filename in os.listdir(err_path):
			file_path = os.path.join(err_path, filename)
			if os.path.isfile(file_path):
				os.remove(file_path)
	else:
		os.makedirs(err_path)

	for i in range(len(errors)):
		img, label, predict = errors[i]
		file_name = f"error_{(i+1):02d}_true{label}_predict{predict}.png"
		plt.imshow(img.squeeze(), cmap='gray')
		plt.axis('off')
		plt.savefig(os.path.join(err_path, file_name))
		plt.close()


def main(params):
	"""
	Main function to execute the training, validation, and testing pipeline.
	"""
	# Initializes model, data loaders, loss function, and optimizer.
	seed = params['seed']
	data_path = '../dataset'

	if torch.cuda.is_available():
		device = torch.device('cuda')
		ratio = params['ratio_cuda']
	else:
		device = torch.device('cpu')
		ratio = params['ratio_cpu']

	print(f"device: {device}")

	# Build the CNN model
	model = build_cnn(
		params['input_channels'], params['conv_channels'], params['kernel_size'],
		params['use_batch_norm'], params['use_dropout'], params['pooling']
	).to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), params['lr'])

	# Load data
	train_loader, val_loader, test_loader = load_data(seed, ratio, data_path)

	# Train and evaluate
	best_model = train(train_loader, val_loader, model, criterion, optimizer)
	test(test_loader, model, best_model, data_path)


if __name__ == '__main__':
	params_ = {
		'seed': 42,
		'ratio_cuda': 0.1,
		'ratio_cpu': 0.01,
		'input_channels': 1,
		'conv_channels': [16, 32],
		'kernel_size': 3,
		'use_batch_norm': True,
		'use_dropout': False,
		'pooling': 'max',
		'lr': 1e-3
	}
	main(params_)
