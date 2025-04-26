import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# -------------------------
# 1. Vocabulary and Tokenization
# -------------------------
class Vocab:
	def __init__(self, tokens, max_size=100000, min_freq=1):
		"""
		Build vocabulary from tokens, keeping those with frequency >= min_freq
		"""
		freq = {}
		for t in tokens:
			freq[t] = freq.get(t, 0) + 1

		# filter and sort by frequency
		items = [t for t, c in freq.items() if c >= min_freq]
		items.sort(key=lambda token: -freq[token])
		items = items[:max_size]

		# special tokens
		self.pad, self.unk = '<pad>', '<unk>'
		self.i2s = [self.pad, self.unk] + items
		self.s2i = {t: i for i, t in enumerate(self.i2s)}

	def encode(self, tokens, max_len=200):
		"""
		Convert list of tokens to list of IDs with left-padding
		"""
		ids = [self.s2i.get(t, self.s2i[self.unk]) for t in tokens[:max_len]]
		if len(ids) < max_len:
			# left-pad with pad token
			pad_count = max_len - len(ids)
			ids = [self.s2i[self.pad]] * pad_count + ids
		return ids


# -------------------------
# 2. Dataset Definition
# -------------------------
class TextDataset(Dataset):
	def __init__(self, texts, labels, vocab, max_len=200):
		self.vocab = vocab
		self.max_len = max_len

		# encode texts and prepare binary labels
		self.texts = [vocab.encode(t.lower().split(), max_len) for t in texts]
		self.labels = np.array(labels, dtype=np.float32)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		x = torch.tensor(self.texts[idx], dtype=torch.long)
		y = torch.tensor(self.labels[idx], dtype=torch.float)
		return x, y


# -------------------------
# 3. Positional Encoding Layer
# -------------------------
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=200):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(0, max_len).unsqueeze(1).float()
		div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(pos * div)
		pe[:, 1::2] = torch.cos(pos * div)
		self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

	def forward(self, x):
		# add positional encoding to embeddings
		x = x + self.pe[:, -x.size(1):, :].to(x.device)
		return x


# -------------------------
# 4. Attention-based Classifier
# -------------------------
class AttentionClassifier(nn.Module):
	def __init__(self, vocab_size, d_model=128, n_head=4, max_len=200):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, d_model)
		self.pos_enc = PositionalEncoding(d_model, max_len)
		self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(d_model, 1)

	def forward(self, x):
		# input x: (batch, seq_len)
		x = self.embed(x)
		x = self.pos_enc(x)

		# causal mask to prevent attending to future tokens
		seq_len = x.size(1)
		mask = torch.triu(
			torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device) * float('-inf'),
			diagonal=1
		)

		attn_out, _ = self.attn(x, x, x, attn_mask=mask)
		attn_out = self.norm(attn_out)

		# use representation of last token for classification
		last_repr = attn_out[:, -1, :]
		logits = self.classifier(self.dropout(last_repr)).squeeze(-1)
		return logits


# -------------------------
# 5. RNN-based Classifier
# -------------------------
class RNNClassifier(nn.Module):
	def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1, rnn_type='gru'):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, emb_dim)
		self.rnn_type = rnn_type

		# choose RNN variant
		if rnn_type == 'rnn':
			self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers, batch_first=True)
		elif rnn_type == 'lstm':
			self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
		elif rnn_type == 'gru':
			self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
		else:
			raise ValueError("rnn_type must be 'rnn', 'lstm', or 'gru'")

		self.classifier = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		x = self.embed(x)
		out, hn = self.rnn(x)

		# handle LSTM hidden state tuple
		if isinstance(hn, tuple):
			hn = hn[0]

		# take last layer's hidden state
		last_hidden = hn[-1]
		logits = self.classifier(last_hidden).squeeze(-1)
		return logits


# -------------------------
# 6. Training and Evaluation
# -------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, patience=3):
	"""
	Train model with early stopping and record training history.
	Returns history dict containing loss and metrics per epoch.
	"""
	best_val_loss = float('inf')
	epochs_no_improve = 0

	history = {
		'train_loss': [],
		'val_loss': [],
		'val_accuracy': [],
		'val_f1': []
	}

	for epoch in range(1, epochs + 1):
		# training step
		train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

		# validation step
		val_metrics = eval_model(model, val_loader, device)
		val_loss = 1.0 - val_metrics['accuracy']  # proxy for loss

		# record history
		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		history['val_accuracy'].append(val_metrics['accuracy'])
		history['val_f1'].append(val_metrics['f1'])

		tqdm.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}")

		# early stopping
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_no_improve = 0
			# torch.save(model.state_dict(), 'best_model.pth')
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				print(f"Early stopping at epoch {epoch}")
				break

	return history


def train_epoch(model, loader, optimizer, criterion, device):
	"""
	Train for one epoch, with tqdm progress bar.
	"""
	model.train()
	total_loss = 0.0

	# wrap loader with tqdm for progress visualization
	for x_batch, y_batch in tqdm(loader, desc='Batch'):
		x_batch, y_batch = x_batch.to(device), y_batch.to(device)
		optimizer.zero_grad()
		logits = model(x_batch)
		loss = criterion(logits, y_batch)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * x_batch.size(0)

	avg_loss = total_loss / len(loader.dataset)
	return avg_loss


def eval_model(model, loader, device):
	model.eval()
	preds, trues = [], []

	with torch.no_grad():
		for x_batch, y_batch in loader:
			x_batch = x_batch.to(device)
			logits = model(x_batch)
			pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
			preds.extend(pred)
			trues.extend(y_batch.numpy())

	# compute metrics
	return {
		'accuracy': accuracy_score(trues, preds),
		'precision': precision_score(trues, preds, zero_division=0),
		'recall': recall_score(trues, preds, zero_division=0),
		'f1': f1_score(trues, preds, zero_division=0)
	}


# -------------------------
# 7. Plotting Utilities
# -------------------------
def plot_training_history(history):
	"""
	Plot train/validation loss, accuracy and F1 over epochs.
	"""
	epochs = range(1, len(history['train_loss']) + 1)

	# Loss plot
	plt.figure()
	plt.plot(epochs, history['train_loss'], label='Train Loss')
	plt.plot(epochs, history['val_loss'], label='Val Loss')
	plt.title('Loss over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

	# Accuracy plot
	plt.figure()
	plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
	plt.title('Validation Accuracy over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.show()

	# F1 score plot
	plt.figure()
	plt.plot(epochs, history['val_f1'], label='Val F1 Score')
	plt.title('Validation F1 Score over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('F1 Score')
	plt.show()


def plot_val_loss_comparison(history1, history2, label1='Model 1', label2='Model 2'):
	"""
	Plot validation loss comparison between two models.
	"""
	epochs = range(1, len(history1['val_loss']) + 1)

	plt.figure()
	plt.plot(epochs, history1['val_loss'], label=label1)
	plt.plot(epochs, history2['val_loss'], label=label2)
	plt.title('Validation Loss Comparison')
	plt.xlabel('Epoch')
	plt.ylabel('Validation Loss')
	plt.legend()
	plt.grid(True)
	plt.show()


# -------------------------
# 8. Main Execution
# -------------------------
def main():
	# load DataFrame df with columns ['text', 'label']
	from load_dataset import df

	# prepare vocabulary
	tokens = [t.lower() for text in df['text'] for t in text.split()]
	vocab = Vocab(tokens, max_size=100000, min_freq=1)

	# prepare dataset and splits
	texts, labels = df['text'].values, df['label'].values
	dataset = TextDataset(texts, labels, vocab, max_len=200)
	n = len(dataset)

	train_size = int(0.8 * n)
	val_size = int(0.1 * n)
	test_size = n - train_size - val_size
	train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=64)
	test_loader = DataLoader(test_ds, batch_size=64)

	# device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# instantiate models
	attn_model = AttentionClassifier(len(vocab.i2s)).to(device)
	rnn_model = RNNClassifier(len(vocab.i2s), rnn_type='rnn').to(device)

	# loss and optimizers
	criterion = nn.BCEWithLogitsLoss()
	attn_opt = torch.optim.Adam(attn_model.parameters(), lr=1e-2)
	rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=1e-2)

	# train and record history
	print('Training Attention model...')
	attn_history = train_model(
		attn_model, train_loader, val_loader, attn_opt, criterion, device,
		epochs=10, patience=11
	)
	plot_training_history(attn_history)

	print('Training RNN model...')
	rnn_history = train_model(
		rnn_model, train_loader, val_loader, rnn_opt, criterion, device,
		epochs=10, patience=11
	)
	plot_training_history(rnn_history)

	# final evaluation on test set
	print('Test Attention:', eval_model(attn_model, test_loader, device))
	print('Test RNN:', eval_model(rnn_model, test_loader, device))

	# visualize validation loss comparison
	plot_val_loss_comparison(attn_history, rnn_history, label1='Attention', label2='RNN')


if __name__ == '__main__':
	main()
