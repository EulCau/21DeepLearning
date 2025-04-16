import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
# import tensordataset

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

df = pd.read_csv('enron_spam_data.csv').drop(columns=['Date'], axis=1).rename(columns={'Spam/Ham': 'label'})
df.dropna(inplace=True)  # 删除缺失值
# 拼接Subject和Message为text
df['text'] = df['Subject'] + ' ' + df['Message'] 
df = df[['text', 'label']]
# Convert text labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(df.head())

# Analyze text length
text_lengths = df['text'].apply(lambda x: len(x.split()))

# Display basic statistics about text lengths
print("Text length statistics:")
print(f"Mean: {text_lengths.mean():.2f}")
print(f"Median: {text_lengths.median():.2f}")
print(f"Min: {text_lengths.min()}")
print(f"Max: {text_lengths.max()}")
print(f"95th percentile: {text_lengths.quantile(0.95):.2f}")

# Compare length between spam and ham messages
spam_lengths = text_lengths[df['label'] == 1]
ham_lengths = text_lengths[df['label'] == 0]

print(f"\nSpam messages (count: {len(spam_lengths)}):")
print(f"Mean length: {spam_lengths.mean():.2f}")

print(f"\nHam messages (count: {len(ham_lengths)}):")
print(f"Mean length: {ham_lengths.mean():.2f}")