# Graph Autoencoder (GAE)

**中文标题:** 图自编码器（GAE）

**难度:** Hard

**函数名:** `gae`

## 英文描述

Implement a Graph Autoencoder that learns node embeddings via a two-layer GCN encoder and reconstructs the adjacency matrix with an inner-product decoder.

**Signature:** `gae(A, X) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F)

**Returns:** reconstructed adjacency matrix `A_hat` of shape (N, N), with values in [0, 1] (apply sigmoid)

**Architecture:**
1. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}` where D is the degree matrix of `A + I`
2. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)` with `W1` shape (F, 32)
3. GCN Layer 2: `Z = A_norm @ H @ W2` with `Z` shape (N, 16), `W2` shape (32, 16)
4. Inner-product decoder: `A_hat = sigmoid(Z @ Z^T)`

**Constraints:**
- Initialize `W1` and `W2` with `torch.manual_seed(0)` then `torch.randn(...) * 0.1` (W1 first, then W2)
- `W1` shape: (F, 32), `W2` shape: (32, 16)
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` and `torch.sigmoid`

## 中文描述

实现图自编码器 (GAE)，通过两层 GCN 编码器学习节点嵌入，并用内积解码器重建邻接矩阵。

**签名:** `gae(A, X) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F)

**返回:** 重建的邻接矩阵 `A_hat`，形状 (N, N)，值在 [0, 1] 之间（应用 sigmoid）

**架构:**
1. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`，其中 D 是 `A + I` 的度矩阵
2. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`，`W1` 形状 (F, 32)
3. GCN 第二层：`Z = A_norm @ H @ W2`，`Z` 形状 (N, 16)，`W2` 形状 (32, 16)
4. 内积解码器：`A_hat = sigmoid(Z @ Z^T)`

**约束:**
- 用 `torch.manual_seed(0)` 初始化 `W1` 和 `W2`，然后依次 `torch.randn(...) * 0.1`（先 W1 后 W2）
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 和 `torch.sigmoid`

## 提示

```
1. `A_hat = A + I` (add self-loops)
2. `D_inv_sqrt = diag(A_hat.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt`
4. `W1 = randn(F, 32) * 0.1`, `W2 = randn(32, 16) * 0.1`
5. `H = relu(A_norm @ X @ W1)`
6. `Z = A_norm @ H @ W2`
7. `return sigmoid(Z @ Z.T)`
```

## 参考答案

```python
def gae(A, X):
    import torch
    N = A.shape[0]
    F = X.shape[1]
    torch.manual_seed(0)
    W1 = torch.randn(F, 32) * 0.1
    W2 = torch.randn(32, 16) * 0.1
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    H = torch.relu(A_norm @ X @ W1)
    Z = A_norm @ H @ W2
    return torch.sigmoid(Z @ Z.T)
```

