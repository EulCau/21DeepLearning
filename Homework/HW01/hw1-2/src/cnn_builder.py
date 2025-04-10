import torch.nn as nn


class AutoCNN(nn.Module):
	def __init__(self, input_channels, conv_channels, kernel_size=3,
				 use_batch_norm=True, use_dropout=False, pooling='max', num_classes=10):
		super(AutoCNN, self).__init__()

		layers = []
		in_channels = input_channels

		for out_channels in conv_channels:
			# 卷积层
			layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
			if use_batch_norm:
				layers.append(nn.BatchNorm2d(out_channels))
			layers.append(nn.ReLU(inplace=True))
			# 池化层
			if pooling == 'max':
				layers.append(nn.MaxPool2d(kernel_size=2))
			elif pooling == 'avg':
				layers.append(nn.AvgPool2d(kernel_size=2))
			else:
				raise ValueError("pooling must be 'max' or 'avg'")
			if use_dropout:
				layers.append(nn.Dropout(0.25))

			in_channels = out_channels

		self.conv = nn.Sequential(*layers)

		# 输入图像是 28x28，经过多次2倍池化后的尺寸
		pool_factor = 2 ** len(conv_channels)
		final_size = 28 // pool_factor
		fc_input_dim = conv_channels[-1] * final_size * final_size

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(fc_input_dim, 128),
			nn.ReLU(),
			nn.Dropout(0.5) if use_dropout else nn.Identity(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
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
