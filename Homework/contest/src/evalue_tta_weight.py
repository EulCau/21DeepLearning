import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from cache_features import extract_features
from inference import InferenceDataset
from model import AIDetectorModel
from tta_utils import apply_tta

def evaluate_tta_weights(model_ckpt_paths, tokenizer_name, val_jsonl_path, tta_modes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	with open(val_jsonl_path, 'r', encoding="utf-8") as f:
		val_data = [json.loads(line) for line in f]

	true_labels = [item["label"] for item in val_data]
	n_samples = len(val_data)

	all_preds = {mode: np.zeros(n_samples) for mode in tta_modes}

	for tta_mode in tta_modes:
		print(f"Evaluating TTA mode: {tta_mode}")
		aug_texts = [apply_tta(item["text"], tta_mode) for item in val_data]

		# 构造临时文件或直接构造 Dataset
		tmp_aug_path = "../data/temp_val_aug.jsonl"
		with open(tmp_aug_path, 'w', encoding="utf-8") as f:
			for text in aug_texts:
				feats = extract_features(text)
				f.write(json.dumps({"text": text, "features": feats}, ensure_ascii=False) + '\n')

		dataset = InferenceDataset(tmp_aug_path, tokenizer_name)
		dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

		fold_preds = []

		for ckpt_path in model_ckpt_paths:
			print(f"  Loading model from {ckpt_path}")
			model = AIDetectorModel.load_from_checkpoint(ckpt_path, model_name=tokenizer_name, feature_dim=14, lr=2e-5)
			model.eval()
			model.to(device)

			preds = []
			with torch.no_grad():
				for batch in dataloader:
					input_ids = batch["input_ids"].to(device)
					attention_mask = batch["attention_mask"].to(device)
					features = batch["features"].to(device)

					logits = model(input_ids, attention_mask, features)
					probs = torch.softmax(logits, dim=1)[:, 1]
					preds.extend(probs.cpu().numpy().tolist())
			fold_preds.append(preds)

		# 平均 fold 预测
		avg_preds = np.mean(fold_preds, axis=0)
		all_preds[tta_mode] = avg_preds

		# 评估指标
		auc = roc_auc_score(true_labels, avg_preds)
		pred_labels = (avg_preds >= 0.5).astype(int)
		acc = accuracy_score(true_labels, pred_labels)
		print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}")

	# 基于 AUC 归一化权重
	aucs = np.array([roc_auc_score(true_labels, all_preds[mode]) for mode in tta_modes])
	weights = aucs / aucs.sum()

	print("TTA 模式权重分配（基于验证集 AUC）:")
	for mode, w in zip(tta_modes, weights):
		print(f"  {mode}: {w:.4f}")

	return weights


def run_inference_tta_with_weights(model_ckpt_paths, tokenizer_name, test_jsonl_path, output_txt_path, tta_modes, tta_weights):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	with open(test_jsonl_path, 'r', encoding="utf-8") as f:
		raw_data = [json.loads(line) for line in f]

	all_preds = []

	for tta_mode in tta_modes:
		print(f"--- TTA mode: {tta_mode} ---")
		aug_texts = [apply_tta(item["text"], tta_mode) for item in raw_data]

		tmp_aug_path = "../data/temp_aug.jsonl"
		with open(tmp_aug_path, 'w', encoding="utf-8") as f:
			for text in aug_texts:
				feats = extract_features(text)
				f.write(json.dumps({"text": text, "features": feats}, ensure_ascii=False) + '\n')

		dataset = InferenceDataset(tmp_aug_path, tokenizer_name)
		dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

		fold_preds = []
		for ckpt_path in model_ckpt_paths:
			print(f"    Loading model from {ckpt_path}")
			model = AIDetectorModel.load_from_checkpoint(ckpt_path, model_name=tokenizer_name, feature_dim=14, lr=2e-5)
			model.eval()
			model.to(device)

			preds = []
			with torch.no_grad():
				for batch in dataloader:
					input_ids = batch["input_ids"].to(device)
					attention_mask = batch["attention_mask"].to(device)
					features = batch["features"].to(device)

					logits = model(input_ids, attention_mask, features)
					probs = torch.softmax(logits, dim=1)[:, 1]
					preds.extend(probs.cpu().numpy().tolist())
			fold_preds.append(preds)

		avg_preds = np.mean(fold_preds, axis=0)
		all_preds.append(avg_preds)

	# 用权重加权融合
	weighted_preds = np.zeros(len(raw_data))
	for w, preds in zip(tta_weights, all_preds):
		weighted_preds += w * preds
	weighted_preds /= sum(tta_weights)

	final_labels = [1 if p >= 0.5 else 0 for p in weighted_preds]

	print(f"Saving predictions to {output_txt_path}")
	with open(output_txt_path, 'w', encoding="utf-8") as f:
		for label in final_labels:
			f.write(f"{label}\n")
	print("Done.")


if __name__ == "__main__":
	model_ckpt_paths_ = [
		"../result/checkpoints/fold1-best-checkpoint.ckpt",
		"../result/checkpoints/fold2-best-checkpoint.ckpt",
		"../result/checkpoints/fold3-best-checkpoint.ckpt",
		"../result/checkpoints/fold4-best-checkpoint.ckpt",
		"../result/checkpoints/fold5-best-checkpoint.ckpt"
	]

	tokenizer_name_ = "roberta-base"
	test_jsonl_path_ = "../data/test.jsonl"
	output_txt_path_ = "../result/submit.txt"

	tta_modes_ = ["lower", "shuffle_sent", "drop_stopwords", "char_noise", "synonym_replace", "shuffle_word", "shuffle_char"]
	weights = evaluate_tta_weights(model_ckpt_paths_, tokenizer_name_, val_jsonl_path_, tta_modes_)

	# Step2: 在测试集上用权重融合预测
	run_inference_tta_with_weights(model_ckpt_paths_, tokenizer_name_, test_jsonl_path_, output_txt_path_, tta_modes_,
								   weights)

