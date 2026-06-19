# Link Prediction

**中文标题:** 链接预测

**难度:** Hard

**函数名:** `link_prediction`

## 英文描述

Implement a link prediction model using a two-layer GCN encoder that predicts edge existence probabilities.

**Signature:** `link_prediction(A, X, edges) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F)
- `edges` — (M, 2) long tensor of node pairs to predict

**Returns:** (M,) tensor of probabilities in [0, 1], where each value is the predicted likelihood of an edge existing between the corresponding node pair

**Algorithm:**
1. Initialize weights with `torch.manual_seed(0)`: `W1 = randn(F, 16) * 0.1`, then `W2 = randn(16, 8) * 0.1`
2. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}`
3. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)`
4. GCN Layer 2: `Z = A_norm @ H @ W2` (no activation)
5. For each edge (i, j): `score = sigmoid(Z[i] · Z[j])`
6. Return all scores as (M,) tensor

**Constraints:**
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` and `torch.sigmoid`

## 中文描述

实现基于两层 GCN 编码器的链接预测模型，预测节点对之间存在边的概率。

**签名:** `link_prediction(A, X, edges) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F)
- `edges` — (M, 2) 长整型张量，表示待预测的节点对

**返回:** (M,) 概率张量，值在 [0, 1] 之间，每个值表示对应节点对之间存在边的预测概率

**算法:**
1. 用 `torch.manual_seed(0)` 初始化权重：`W1 = randn(F, 16) * 0.1`，然后 `W2 = randn(16, 8) * 0.1`
2. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`
3. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`
4. GCN 第二层：`Z = A_norm @ H @ W2`（无激活函数）
5. 对每条边 (i, j)：`score = sigmoid(Z[i] · Z[j])`
6. 返回所有分数组成的 (M,) 张量

**约束:**
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 和 `torch.sigmoid`

## 提示

```
1. `A_tilde = A + I`
2. `D_inv_sqrt = diag(A_tilde.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
4. Two-layer GCN: H = relu(A_norm @ X @ W1), Z = A_norm @ H @ W2
5. Score each edge: `sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))`
```

## 参考答案

```python
def link_prediction(A, X, edges):
    import torch
    N = A.shape[0]
    F = X.shape[1]
    torch.manual_seed(0)
    W1 = torch.randn(F, 16) * 0.1
    W2 = torch.randn(16, 8) * 0.1
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    H = torch.relu(A_norm @ X @ W1)
    Z = A_norm @ H @ W2
    scores = torch.sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))
    return scores
```

