# GraphSAGE Layer

**中文标题:** GraphSAGE 层

**难度:** Hard

**函数名:** `graphsage_layer`

## 英文描述

Implement a GraphSAGE layer that aggregates neighbor features via mean pooling, concatenates with self features, and applies a linear transformation with L2 normalization.

**Signature:** `graphsage_layer(A, X, W, k=None) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (2*F_in, F_out)
- `k` — if not None, sample at most k neighbors per node (use `torch.manual_seed(42)` before sampling for determinism)

**Returns:** (N, F_out) tensor, L2-normalized along the feature dimension

**Algorithm:**
1. For each node i:
   - Get neighbor indices where A[i] > 0
   - If k is not None and num_neighbors > k: set `torch.manual_seed(42)`, use `torch.randperm(num_neighbors)[:k]` to sample
   - Compute mean of neighbor features (zero vector if no neighbors)
2. Concatenate [X, neighbor_agg] along feature dim to get (N, 2*F_in)
3. Apply linear transform: `out = ReLU(concat @ W)`
4. L2 normalize: `out = out / ||out||_2` per row (clamp norm min to 1e-8)

**Constraints:**
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` for activation
- Sampling must be deterministic with `torch.manual_seed(42)` called immediately before each `randperm`

## 中文描述

实现 GraphSAGE 层：通过均值池化聚合邻居特征，与自身特征拼接后进行线性变换，最后做 L2 归一化。

**签名:** `graphsage_layer(A, X, W, k=None) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (2*F_in, F_out)
- `k` — 若不为 None，每个节点最多采样 k 个邻居（采样前调用 `torch.manual_seed(42)` 保证确定性）

**返回:** (N, F_out) 张量，沿特征维度做 L2 归一化

**算法:**
1. 对每个节点 i：
   - 获取 A[i] > 0 的邻居索引
   - 若 k 不为 None 且邻居数 > k：设置 `torch.manual_seed(42)`，用 `torch.randperm(num_neighbors)[:k]` 采样
   - 计算邻居特征均值（无邻居则为零向量）
2. 沿特征维拼接 [X, neighbor_agg] 得到 (N, 2*F_in)
3. 线性变换：`out = ReLU(concat @ W)`
4. L2 归一化：`out = out / ||out||_2`（每行归一化，范数下限 clamp 为 1e-8）

**约束:**
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 作为激活函数
- 采样必须确定性：每次 `randperm` 前立即调用 `torch.manual_seed(42)`

## 提示

```
1. Loop over nodes, gather neighbors from A[i].nonzero()
2. If k sampling needed: `torch.manual_seed(42)`, `perm = torch.randperm(n_neighbors)[:k]`
3. Mean-pool neighbor features (or zeros if isolated)
4. `concat = torch.cat([X, agg], dim=1)` → (N, 2*F_in)
5. `out = relu(concat @ W)`
6. `out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)`
```

## 参考答案

```python
def graphsage_layer(A, X, W, k=None):
    import torch
    N = A.shape[0]
    F_in = X.shape[1]
    agg_list = []
    for i in range(N):
        neighbors = (A[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) == 0:
            agg_list.append(torch.zeros(F_in))
        else:
            if k is not None and len(neighbors) > k:
                torch.manual_seed(42)
                perm = torch.randperm(len(neighbors))[:k]
                neighbors = neighbors[perm]
            agg_list.append(X[neighbors].mean(dim=0))
    agg = torch.stack(agg_list)
    concat = torch.cat([X, agg], dim=1)
    out = torch.relu(concat @ W)
    out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return out
```

