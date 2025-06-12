import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from features import stylometry_features


class AIDetectorDataset(Dataset):
	def __init__(self, jsonl_path, tokenizer_name, max_length=512, mode='train'):
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
		self.max_length = max_length
		self.mode = mode

		self.samples = []
		with open(jsonl_path, 'r', encoding='utf-8') as f:
			for line in f:
				item = json.loads(line)
				text = item['text']
				label = item.get('label', -1)  # label 可能没有（测试集）
				features = self.extract_features(text)
				self.samples.append((text, label, features))

	@staticmethod
	def extract_features(text):
		# Simple features
		length = len(text)
		num_hashes = text.count('#')
		num_urls = text.count('http')
		num_lists = text.count('\n- ') + text.count('\n1.') + text.count('\n2.')
		simple_feats = [length, num_hashes, num_urls, num_lists]

		# Stylometry features
		styl_feats = stylometry_features(text)

		return simple_feats + styl_feats

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
