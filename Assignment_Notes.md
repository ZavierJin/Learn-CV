# 练习记录

## Reference

https://github.com/Zhang-Each/Awesome-CS-Course-Learning-Notes/tree/master/Stanford-CS231N-NeuralNetwork%26DL

https://github.com/lightaime/cs231n

https://github.com/jariasf/CS231n

## KNN

训练：将训练集的特征和标签储存

预测：计算输入与训练集的距离，根据距离最小的前k个标签，选择标签最多的作为预测结果

将图像利用numpy完全向量化再进行运算能够显著提高运算速度。

```python
dists = np.sqrt(np.sum(X ** 2, axis=-1).reshape(-1, 1) + np.sum(self.X_train ** 2, axis=-1) - 2 * np.matmul(X, self.X_train.T))
```

| 向量化程度  | 两层循环 | 一层循环 | 完全向量化 |
| ----------- | -------- | -------- | ---------- |
| 运行时间(s) | 41.97    | 32.64    | 0.63       |

超参的调整：测试最优的k值选取。使用5交叉验证。

![knn](D:\zju\study\08_大四下\课外阅读\Learn-CV\Image\knn.png)

## SVM损失函数

```
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
```

| 左对齐 | 右对齐 | 居中对齐 |
| :----- | ------ | -------- |
|        |        |          |

