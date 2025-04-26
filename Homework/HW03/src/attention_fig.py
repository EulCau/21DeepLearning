import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from rnn_attention import PositionalEncoding

# 设置参数
seq_len = 10
vocab_size = 50
d_model = 16
n_head = 2

# 构建模型组件
embed = nn.Embedding(vocab_size, d_model)
pos_enc = PositionalEncoding(d_model, max_len=seq_len)
attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

# 构造一条序列，输入为token编号
x = torch.arange(0, seq_len).unsqueeze(0)  # shape: [1, seq_len]
x_embed = embed(x)
x_embed = pos_enc(x_embed)

# 自回归 attention mask（下三角为True，其余为False）
mask = torch.tril(torch.ones(seq_len, seq_len)).eq(0)

# 前向传播获取注意力权重
with torch.no_grad():
    out, attn_weights = attn(x_embed, x_embed, x_embed, attn_mask=mask)

# 可视化第一个注意力头的注意力矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(
    attn_weights[0], cmap="viridis",
    xticklabels=[str(i) for i in range(seq_len)],
    yticklabels=[str(i) for i in range(seq_len)]
)
plt.title("Self-Attention Weights (Head 0)")
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.show()
