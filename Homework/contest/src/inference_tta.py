import json

import torch
from torch.utils.data import DataLoader

from cache_features import extract_features
from inference import InferenceDataset
from model import AIDetectorModel
from tta_utils import apply_tta


def run_inference_tta(model_ckpt_paths, tokenizer_name, test_jsonl_path, output_txt_path, tta_modes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	with open(test_jsonl_path, 'r', encoding="utf-8") as f:
		raw_data = [json.loads(line) for line in f]

	# 每种增强方式都跑一遍模型推理
	all_preds = []

	for tta_mode in tta_modes:
		print(f"--- TTA mode: {tta_mode} ---")
		# 创建增强文本
		aug_texts = [apply_tta(item["text"], tta_mode) for item in raw_data]

		# 写临时增强文件
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

		# 当前 TTA 模式下的平均输出
		avg_preds = [sum(preds[i] for preds in fold_preds) / len(fold_preds) for i in range(len(raw_data))]
		all_preds.append(avg_preds)

	# 所有 TTA 模式融合
	final_preds = [sum(preds[i] for preds in all_preds) / len(all_preds) for i in range(len(raw_data))]
	final_labels = [1 if p >= 0.5 else 0 for p in final_preds]

	# Save results
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

	tta_modes_ = ["lower", "shuffle_sent", "drop_stopwords"]  # 可选增强模式
	run_inference_tta(model_ckpt_paths_, tokenizer_name_, test_jsonl_path_, output_txt_path_, tta_modes_)
