import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import f1_score


class AIDetectorModel(pl.LightningModule):
	def __init__(self, model_name, feature_dim, lr=2e-5):
		super().__init__()
		self.save_hyperparameters()

		self.backbone = AutoModel.from_pretrained(model_name)
		hidden_size = self.backbone.config.hidden_size

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size + feature_dim, 512),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(512, 2)
		)

		self.loss_fn = nn.CrossEntropyLoss()
		self.lr = lr

		# 用于保存每个 batch 的预测与标签, 供 epoch_end 使用
		self.validation_step_outputs = []

	def forward(self, input_ids, attention_mask, features):
		outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

		x = torch.cat([pooled_output, features], dim=1)
		logits = self.classifier(x)
		return logits

	def training_step(self, batch, batch_idx):
		logits = self(
			batch["input_ids"],
			batch["attention_mask"],
			batch["features"]
		)
		loss = self.loss_fn(logits, batch["label"])
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		logits = self(
			batch["input_ids"],
			batch["attention_mask"],
			batch["features"]
		)
		loss = self.loss_fn(logits, batch["label"])

		preds = torch.argmax(logits, dim=1)
		labels = batch["label"]

		self.log("val_loss", loss, prog_bar=True)

		# 保存 CPU 上的预测和标签
		self.validation_step_outputs.append({
			"preds": preds.detach().cpu(),
			"labels": labels.detach().cpu()
		})

	def on_validation_epoch_end(self):
		preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
		labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

		f1 = f1_score(labels, preds)
		self.log("val_f1", f1, prog_bar=True)

		# 清空缓存
		self.validation_step_outputs.clear()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer
