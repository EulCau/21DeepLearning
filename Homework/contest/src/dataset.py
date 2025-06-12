import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AIDetectorDataset(Dataset):
	def __init__(self, cached_jsonl_path, tokenizer_name, max_length=512, mode='train'):
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
		self.max_length = max_length
		self.mode = mode

		self.samples = []
		with open(cached_jsonl_path, 'r') as f:
			for line in f:
				item = json.loads(line)
				self.samples.append((item['text'], item.get('label', -1), item['features']))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		text, label, features = self.samples[idx]

		encoding = self.tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=self.max_length,
			return_tensors='pt'
		)

		return {
			'input_ids': encoding['input_ids'].squeeze(0),
			'attention_mask': encoding['attention_mask'].squeeze(0),
			'features': torch.tensor(features, dtype=torch.float),
			'label': torch.tensor(label, dtype=torch.long) if label != -1 else -1
		}
