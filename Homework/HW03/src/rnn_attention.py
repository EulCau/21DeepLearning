import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split


# -------------------------
# 1. Vocabulary and Tokenization
# -------------------------
class Vocab:
	def __init__(self, tokens, max_size=100000, min_freq=1):
		# count frequencies
		freq = {}
		for t in tokens:
			freq[t] = freq.get(t, 0) + 1
		# keep those above min_freq
		items = [t for t, c in freq.items() if c >= min_freq]
		# sort by freq
		items.sort(key=lambda token: -freq[token])
		# limit size
		items = items[: max_size]
		# add special tokens
		self.unk, self.pad = '<unk>', '<pad>'
		self.i2s = [self.pad, self.unk] + items
		self.s2i = {t: i for i, t in enumerate(self.i2s)}

	def encode(self, tokens, max_len=200):
		# map tokens to ids, pad/truncate
		ids = [self.s2i.get(t, self.s2i[self.unk]) for t in tokens[:max_len]]
		if len(ids) < max_len:
			ids = [self.s2i[self.pad]] * (max_len - len(ids)) + ids  # left-pad
		return ids


# -------------------------
# 2. Dataset
# -------------------------
class TextDataset(Dataset):
	def __init__(self, texts, labels, vocab, max_len=200):
		self.vocab = vocab
		self.max_len = max_len
		self.texts = [vocab.encode(t.lower().split(), max_len) for t in texts]
		self.labels = (labels == 1).astype(np.float32)  # ensure binary float labels

	def __len__(self): return len(self.labels)

	def __getitem__(self, idx):
		return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


# -------------------------
# 3. Positional Encoding
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
		x = x + self.pe[:, -x.size(1):, :].to(x.device)
		return x


# -------------------------
# 4. Attention-based Model
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
		# x: (batch, seq)
		x = self.embed(x)
		x = self.pos_enc(x)

		# self-attention (query,key,value all x)
		mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).to(x.device).bool()
		attn_out, _ = self.attn(x, x, x, attn_mask=mask)
		# take last token's representation
		last = attn_out[:, -1, :]
		return self.classifier(last).squeeze(-1)


# -------------------------
# 5. RNN-based Model
# -------------------------
class RNNClassifier(nn.Module):
	def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1, rnn_type='rnn'):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, emb_dim)
		self.rnn_type = rnn_type
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
		# 处理LSTM的输出（hn为元组）
		if isinstance(hn, tuple): hn = hn[0]
		last = hn[-1]  # 取最后一层的隐藏状态
		return self.classifier(last).squeeze(-1)


# -------------------------
# 6. Training and Evaluation Utilities
# -------------------------

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, patience=3):
	best_val_loss = float('inf')
	epochs_no_improve = 0
	for epoch in range(epochs):
		train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
		val_metrics = eval_model(model, val_loader, device)
		val_loss = 1 - val_metrics['accuracy']
		print(f'Epoch {epoch}: loss={train_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1']:.4f}')

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_no_improve = 0
			torch.save(model.state_dict(), 'best_model.pth')
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				print(f"Early stopping at epoch {epoch}")
				break
	# 输出训练日志...


def train_epoch(model, loader, optimizer, criterion, device):
	model.train()
	total_loss = 0
	for x, y in loader:
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		logits = model(x)
		loss = criterion(logits, y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * x.size(0)
	return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
	model.eval()
	preds, trues = [], []
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			logits = model(x)
			pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
			preds.extend(pred)
			trues.extend(y.numpy())
	# compute metrics
	return {
		'accuracy': accuracy_score(trues, preds),
		'precision': precision_score(trues, preds, zero_division=0),
		'recall': recall_score(trues, preds, zero_division=0),
		'f1': f1_score(trues, preds, zero_division=0)
	}

# -------------------------
# 7. Putting It All Together
# -------------------------
def main():
	from load_dataset import df

	# build vocabulary
	tokens = [t.lower() for text in df['text'] for t in text.split()]
	vocab = Vocab(tokens, max_size=50000, min_freq=2)

	# split data
	texts = df['text'].values
	labels = df['label'].values
	labels = (labels == 1).astype(np.float32)
	dataset = TextDataset(texts, labels, vocab, max_len=200)
	n = len(dataset)
	train_len = int(0.8 * n)
	val_len = int(0.1 * n)
	test_len = n - train_len - val_len
	train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
	train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=64)
	test_loader = DataLoader(test_ds, batch_size=64)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# instantiate models
	attn_model = AttentionClassifier(len(vocab.i2s)).to(device)
	rnn_model = RNNClassifier(len(vocab.i2s)).to(device)

	# criterion and optimizers
	criterion = nn.BCEWithLogitsLoss()
	attn_opt = torch.optim.Adam(attn_model.parameters(), lr=1e-3)
	rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)

	epochs = 10

	# train attention model
	print('Training Attention model...')
	train_model(attn_model, train_loader, val_loader, attn_opt, criterion, device, epochs)

	# train RNN model
	print('Training RNN model...')
	train_model(rnn_model, train_loader, val_loader, rnn_opt, criterion, device, epochs)

	# final evaluation on test set
	print('Test Attention:', eval_model(attn_model, test_loader, device))
	print('Test RNN:', eval_model(rnn_model, test_loader, device))


if __name__ == '__main__':
	main()
