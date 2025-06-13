# AI Text Detector 项目说明文档

## 项目简介

本项目旨在利用预训练语言模型与人工设计特征融合的方式, 对文本数据进行 AI 生成检测. 基础模型为 `roberta-base`, 辅以 14 维风格和统计特征输入. 模型通过 5 折交叉验证 (K-Fold) 训练, 并支持推理阶段的 Test-Time Augmentation (TTA) 以提升泛化能力.

---

## 文件结构与功能说明

```
├── data/                       # 存放数据文件
├── result/                     # 存放模型权重与输出
├── checkpoints/                # 每折模型保存路径
├── src/                        # 存放源代码
|   ├── kfold_train.py          # K-Fold 训练主程序
|   ├── model.py                # 模型定义 (RoBERTa + 特征融合)
|   ├── dataset.py              # 训练数据集定义
|   ├── inference.py            # 推理脚本(无 TTA)
|   ├── inference_tta.py        # 推理脚本(有 TTA)
|   ├── cache_features.py       # 文本风格和统计特征提取模块
|   └── tta_utils.py            # TTA 方法定义与应用
|
└── requirements.txt            # 依赖文件
```

---

## 模型结构说明

文件: `model.py`

### 主体结构

* 主干网络: `roberta-base`, 提取 \[CLS] 表征 (`last_hidden_state[:, 0, :]`). 
* 特征融合: 拼接 handcrafted features (14 维) 与 RoBERTa 输出. 
* 分类器: `Linear(768 + 14) → ReLU → Dropout → Linear(512 → 2)`
* 损失函数: `CrossEntropyLoss`
* 评价指标: `F1-score` (用于 early stopping 和 checkpoint)

---

## 特征工程

文件: `cache_features.py`

共提取 14 个特征, 分为两类:

### Simple Features (4维)

* 文本长度
* `#` 数量
* URL 数量
* 列表条目数 (如“1.”、“- ”等)

### Stylometric Features (10维)

* 词汇多样性
* 平均词长
* 停用词比例
* 平均句长
* 标点比例
* Flesch 阅读难度
* 名词/动词/形容词/副词比例 (POS分布)

---

## 数据处理流程

### 训练前预处理

运行 `cache_features.py`, 使训练过程中不需要反复求特征值:

```bash
python cache_features.py --input ../data/train.jsonl --output ../data/cached_train.jsonl --workers 8
python cache_features.py --input ../data/test.jsonl --output ../data/cached_test.jsonl --workers 8
```

### 数据集定义

文件: `dataset.py`

* 支持 `train/val/test` 模式
* 使用 HuggingFace `AutoTokenizer` 分词
* 返回字段: `input_ids`, `attention_mask`, `features`, `label`

---

## 模型训练

文件: `kfold_train.py`

采用 `StratifiedKFold(n_splits=5)` 进行训练, 确保各折类别分布一致.

* 每一折保存一个 checkpoint: `fold{i}-best-checkpoint.ckpt`
* 最佳模型依据验证集 `val_f1` 保存
* 提供早停机制 (`patience=3`), 最多训练 5 个 epoch

日志显示每一折最终 F1-score 在 `0.985 ~ 0.994` 之间, 表现稳定.

---

## 推理方式

### 1. 不使用 TTA

文件: `inference.py`
函数: `run_inference(...)`

步骤:

1. 加载 5 个 fold 的 checkpoint
2. 对 test 数据进行特征提取 (需提前缓存)
3. 计算每个模型的预测概率并求平均
4. 预测结果写入 `submit.txt`

调用示例:

```bash
python inference.py
```

---

### 2. 使用 TTA

文件: `inference.py`
函数: `run_inference_tta_with_weights(...)`

TTA 方法定义在 `tta_utils.py` 中, 包含以下 7 种策略:

* `lower`: 小写化
* `shuffle_sent`: 打乱句子顺序
* `shuffle_words`: 打乱句内词序
* `drop_stopwords`: 删除停用词
* `drop_sentence`: 随机删除句子
* `replace_synonym`: 同义词替换
* `char_noise`: 字符扰动 (删除/重复/交换)

### 权重设定

原始方法中使用 `evaluate_tta_weights(...)` 自动计算各模式在验证集上的 AUC 并归一化作为权重. 为加快推理, 最终权重 **已手动写死**, 如下:

```python
tta_modes_ = ["lower", "shuffle_sent", "shuffle_words", "drop_stopwords", "drop_sentence", "replace_synonym", "char_noise"]
weights_   = [1.0,     1.0,            0.47,            1.0,              1.0,             0.36,              1.0]
```

这些权重在验证集上评估合理, 避免了 TTA 之间性能相差过大的影响.

### 调用方式

```bash
python inference_tta.py
```

---

## 使用说明总结

### 环境依赖

```bash
pip install -r requirements.txt
```

首次运行需要下载资源:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
python -m spacy download en_core_web_sm
```

### 全流程指令汇总

#### Step 1. 特征提取

```bash
python contest/src/cache_features.py --input ../data/train.jsonl --output ../data/cached_train.jsonl --workers 8
```

#### Step 2. 训练模型

```bash
python kfold_train.py
```

#### Step 3. 对 test 数据做特征

```bash
python cache_features.py --input ../data/test.jsonl --output ../data/cached_test.jsonl --workers 8
```

#### Step 4. 推理 (无 TTA)

```bash
python inference.py
```

#### Step 5. 推理 (含 TTA)

```bash
python inference_tta.py
```

---

## 成果总结

| 模式     | 提交分数   |
| ------ | ------ |
| 无 TTA  | 0.5600 |
| 使用 TTA | 0.7989 |

**提升幅度显著, TTA 对于提升模型鲁棒性具有关键作用.**
