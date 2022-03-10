# 练习记录

## KNN

将图像利用numpy完全向量化再进行运算能够显著提高运算速度。

```python
dists = np.sqrt(np.sum(X ** 2, axis=-1).reshape(-1, 1) + np.sum(self.X_train ** 2, axis=-1) - 2 * np.matmul(X, self.X_train.T))
```

| 向量化程度  | 两层循环 | 一层循环 | 完全向量化 |
| ----------- | -------- | -------- | ---------- |
| 运行时间(s) | 35.7     | 25.6     | 0.17       |

## SVM损失函数
