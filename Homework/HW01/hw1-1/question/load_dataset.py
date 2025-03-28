from sklearn.datasets import fetch_openml
import pandas as pd

# 自动下载数据集（约50KB）
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# 查看数据
print(f"特征字段: {boston.feature_names}")
print(f"样本数: {len(df)}")
df.head()
