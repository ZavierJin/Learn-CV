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

## SVM

- 使用多分类SVM作为损失函数，单个样本的损失函数$$\displaystyle L_{i}=\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+\Delta\right)=\sum_{j \neq y_{i}} \max \left(0, w_j^Tx_i-w_{y_i}^Tx_i+\Delta\right) $$，其中输入$x_i$，标签$y_i$，模型输出的score是K维度的$s$，$s$的每一个元素为$s_j$，而$s$中真实标签的元素为$s_{y_i}$。$s_{j}-s_{y_{i}}$即计算其他类对应的元素与真实标签对应元素的score差值，如果超出一定阈值($\Delta$，一般选择1)，就需要累加到损失函数上。
- 未防止过拟合，在平均后添加正则项(这里选用L2范数)，总损失函数为$$\displaystyle L=\frac 1N\sum_{i=1}^NL_{i}+\lambda ||W||^2=\sum_{i=1}^N\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+\Delta\right)+\lambda ||W||_{2}$$。根据奥卡姆剃刀原则，选择简单的W，超参数$\lambda$。
- SVM的梯度推导
  - 首先梯度要根据K个score逐一进行累加(除了正确标签的)
  - 损失是基于margin的，即超出阈值的部分$s_{j}-s_{y_{i}}+\Delta$。下面只推导margin大于0的情况。由于$$\displaystyle \frac{\partial margin}{\partial w_j}=x_j$$，$$\displaystyle \frac{\partial margin}{\partial w_{y_i}}=-x_i$$。因此可以得到梯度公式：$$\displaystyle \frac{\partial L_i}{\partial w_j}=\frac {\partial \sum_{j\not=y_i}\max(0, margin)}{\partial w_j}=x_i $$。