import torch.nn as nn


class AutoCNN(nn.Module):
	"""
	A configurable Convolutional Neural Network for image classification.

	Args:
		input_channels (int): Number of input image channels (e.g., 1 for grayscale).
		conv_channels (list of int): List specifying the number of output channels for each conv layer.
		kernel_size (int): Kernel size for all convolutional layers.
		use_batch_norm (bool): Whether to apply Batch Normalization after each conv layer.
		use_dropout (bool): Whether to apply Dropout after pooling layers and before final linear layer.
		pooling (str): Type of pooling layer to use ('max' or 'avg').
		num_classes (int): Number of output classes.
	"""
	def __init__(
		self, input_channels, conv_channels, kernel_size,
		use_batch_norm, use_dropout, pooling, num_classes
	):
		super(AutoCNN, self).__init__()

		layers: list[nn.Module] = []
		in_channels = input_channels

		# Build convolutional feature extractor
		for out_channels in conv_channels:
			# Convolution layer
			layers.append(
				nn.Conv2d(in_channels, out_channels,
				kernel_size=kernel_size, padding=kernel_size // 2)
			)

			# Optional batch normalization
			if use_batch_norm:
				layers.append(nn.BatchNorm2d(out_channels))

			# Activation
			layers.append(nn.ReLU(inplace=True))

			# Pooling
			if pooling == 'max':
				layers.append(nn.MaxPool2d(kernel_size=2))
			elif pooling == 'avg':
				layers.append(nn.AvgPool2d(kernel_size=2))
			else:
				raise ValueError("pooling must be 'max' or 'avg'")

			# Optional dropout
			if use_dropout:
				layers.append(nn.Dropout(0.25))

			in_channels = out_channels

		self.conv = nn.Sequential(*layers)

		# Compute input dimension to the fully connected layer
		# Assuming input image size is 28x28
		pool_factor = 2 ** len(conv_channels)
		final_size = 28 // pool_factor
		fc_input_dim = conv_channels[-1] * final_size * final_size

		# Build fully connected classifier
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(fc_input_dim, 128),
			nn.ReLU(),
			nn.Dropout(0.5) if use_dropout else nn.Identity(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		"""
		Defines the forward pass of the network.

		Args:
			x (Tensor): Input tensor.

		Returns:
			Tensor: Output logits.
		"""
		x = self.conv(x)
		x = self.fc(x)
		return x


def build_cnn(
		input_channels=1,
		conv_channels=None,
		kernel_size=3,
		use_batch_norm=True,
		use_dropout=False,
		pooling='max',
		num_classes=10
):
	"""
	Builds an AutoCNN instance with the given configuration.

	Args:
		input_channels (int): Number of input image channels.
		conv_channels (list of int): List of output channels for each conv layer.
		kernel_size (int): Kernel size for convolution.
		use_batch_norm (bool): Whether to use Batch Normalization.
		use_dropout (bool): Whether to use Dropout.
		pooling (str): Type of pooling to use ('max' or 'avg').
		num_classes (int): Number of output classes.

	Returns:
		AutoCNN: Instantiated CNN model.
	"""
	if conv_channels is None:
		conv_channels = [16, 32]

	return AutoCNN(
		input_channels=input_channels,
		conv_channels=conv_channels,
		kernel_size=kernel_size,
		use_batch_norm=use_batch_norm,
		use_dropout=use_dropout,
		pooling=pooling,
		num_classes=num_classes
	)
