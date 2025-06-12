import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import AIDetectorModel


# --------------- Dataset ---------------
class InferenceDataset(Dataset):
	def __init__(self, cached_test_jsonl_path, tokenizer_name, max_length=512):
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
		self.max_length = max_length
		self.samples = []

		with open(cached_test_jsonl_path, 'r', encoding='utf-8') as f:
			for line in f:
				item = json.loads(line)
				text = item['text']
				features = item['features']
				self.samples.append((text, features))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		text, features = self.samples[idx]
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
			'features': torch.tensor(features, dtype=torch.float)
		}


# --------------- Inference Function ---------------
def run_inference(model_ckpt_paths, tokenizer_name, cached_test_jsonl_path, output_txt_path):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')

	dataset = InferenceDataset(cached_test_jsonl_path, tokenizer_name)
	dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

	all_preds = []

	for ckpt_path in model_ckpt_paths:
		print(f'Loading model from {ckpt_path}')
		model = AIDetectorModel.load_from_checkpoint(ckpt_path, model_name=tokenizer_name, feature_dim=14, lr=2e-5)
		model.eval()
		model.to(device)

		fold_preds = []
		with torch.no_grad():
			for batch in dataloader:
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				features = batch['features'].to(device)

				logits = model(input_ids, attention_mask, features)
				probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of label=1
				fold_preds.extend(probs.cpu().numpy().tolist())

		all_preds.append(fold_preds)

	# Ensemble: average
	print(f'Ensembling {len(model_ckpt_paths)} models')
	avg_preds = []
	for i in range(len(all_preds[0])):
		avg = sum(preds[i] for preds in all_preds) / len(all_preds)
		avg_preds.append(avg)

	# Convert to label
	final_labels = [1 if p >= 0.5 else 0 for p in avg_preds]

	# Save results
	print(f'Saving predictions to {output_txt_path}')
	with open(output_txt_path, 'w', encoding='utf-8') as f:
		for label in final_labels:
			f.write(f'{label}\n')
	print('Done.')


# --------------- Main ---------------
if __name__ == '__main__':
	# Example: using 5-fold models for ensemble
	model_ckpt_paths_ = [
		'../result/checkpoints/fold1-best-checkpoint.ckpt',
		'../result/checkpoints/fold2-best-checkpoint.ckpt',
		'../result/checkpoints/fold3-best-checkpoint.ckpt',
		'../result/checkpoints/fold4-best-checkpoint.ckpt',
		'../result/checkpoints/fold5-best-checkpoint.ckpt'
	]

	tokenizer_name_ = 'roberta-base'
	cached_test_jsonl_path_ = '../data/cached_test.jsonl'
	output_txt_path_ = '../result/submit.txt'

	run_inference(model_ckpt_paths_, tokenizer_name_, cached_test_jsonl_path_, output_txt_path_)
