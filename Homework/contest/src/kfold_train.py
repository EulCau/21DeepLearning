import nltk
import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("medium")
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from dataset import AIDetectorDataset
from model import AIDetectorModel


def kfold_training(jsonl_path, n_splits=5):
	model_name = 'roberta-base'
	tokenizer_name = model_name
	batch_size = 16
	max_length = 512
	feature_dim = 4 + 10  # 之前 simple feature + stylometry feature

	# Load full dataset (with labels)
	dataset = AIDetectorDataset(jsonl_path, tokenizer_name, max_length=max_length, mode='train')

	# Build label list for stratification
	labels = [item[1] for item in dataset.samples]
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, labels)):
		print(f"\n========== Fold {fold + 1}/{n_splits} ==========")

		train_subset = Subset(dataset, train_idx)
		val_subset = Subset(dataset, val_idx)

		train_loader = DataLoader(
			train_subset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			persistent_workers=True,
			pin_memory=True
		)

		val_loader = DataLoader(
			val_subset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=4,
			persistent_workers=True,
    		pin_memory=True
		)

		model = AIDetectorModel(model_name=model_name, feature_dim=feature_dim, lr=2e-5)

		checkpoint_callback = pl.callbacks.ModelCheckpoint(
			monitor='val_f1',
			mode='max',
			save_top_k=1,
			filename=f'fold{fold + 1}-best-checkpoint'
		)

		early_stop_callback = pl.callbacks.EarlyStopping(
			monitor='val_f1',
			patience=3,
			mode='max'
		)

		trainer = pl.Trainer(
			accelerator='gpu',
			devices=1,
			max_epochs=5,
			precision=16,
			callbacks=[checkpoint_callback, early_stop_callback]
		)

		trainer.fit(model, train_loader, val_loader)

		break


if __name__ == '__main__':
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger_eng')
	kfold_training('../data/train.jsonl', n_splits=5)
