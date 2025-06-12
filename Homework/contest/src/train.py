import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from dataset import AIDetectorDataset
from model import AIDetectorModel


def main():
	model_name = 'roberta-large'
	tokenizer_name = model_name
	batch_size = 16
	max_length = 512
	feature_dim = 4  # 对应 extract_features 里定义的特征数量

	train_dataset = AIDetectorDataset('../data/train.jsonl', tokenizer_name, max_length=max_length, mode='train')
	val_dataset = AIDetectorDataset('data/val.jsonl', tokenizer_name, max_length=max_length, mode='val')

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

	model = AIDetectorModel(model_name=model_name, feature_dim=feature_dim, lr=2e-5)

	checkpoint_callback = ModelCheckpoint(
		monitor='val_f1',
		mode='max',
		save_top_k=1,
		filename='best-checkpoint'
	)

	early_stop_callback = EarlyStopping(
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


if __name__ == '__main__':
	main()
