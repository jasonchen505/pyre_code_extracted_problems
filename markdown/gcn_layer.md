# GCN Layer (Graph Convolution)

**中文标题:** GCN 层（图卷积）

**难度:** Easy

**函数名:** `gcn_layer`

## 英文描述

Implement a single Graph Convolutional Network (GCN) layer with symmetric normalization.

**Signature:** `gcn_layer(A, X, W) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (F_in, F_out)

**Returns:** output node features of shape (N, F_out)

**Algorithm:**
1. Add self-loops: `A_tilde = A + I`
2. Compute degree vector: `D_vec = A_tilde.sum(1)`
3. Symmetric normalization: `D_inv_sqrt = diag(D_vec^{-0.5})`
4. Normalized adjacency: `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. Return `ReLU(A_norm @ X @ W)`

**Constraints:**
- Weights `W` are passed in — do not create them inside the function
- Use `torch.relu` for activation

## 中文描述

实现单层图卷积网络 (GCN)，采用对称归一化。

**签名:** `gcn_layer(A, X, W) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (F_in, F_out)

**返回:** 形状为 (N, F_out) 的输出节点特征

**算法:**
1. 添加自环：`A_tilde = A + I`
2. 计算度向量：`D_vec = A_tilde.sum(1)`
3. 对称归一化：`D_inv_sqrt = diag(D_vec^{-0.5})`
4. 归一化邻接矩阵：`A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. 返回 `ReLU(A_norm @ X @ W)`

**约束:**
- 权重 `W` 由参数传入，不要在函数内部创建
- 使用 `torch.relu` 作为激活函数

## 提示

```
1. `A_tilde = A + torch.eye(N)`
2. `D_vec = A_tilde.sum(1)`
3. `D_inv_sqrt = torch.diag(D_vec.pow(-0.5))`
4. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. `return torch.relu(A_norm @ X @ W)`
```

## 参考答案

```python
def gcn_layer(A, X, W):
    import torch
    N = A.shape[0]
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return torch.relu(A_norm @ X @ W)
```

