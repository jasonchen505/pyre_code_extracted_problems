# GAT Layer (Graph Attention)

**中文标题:** GAT 层（图注意力）

**难度:** Medium

**函数名:** `gat_layer`

## 英文描述

Implement a single-head Graph Attention Network (GAT) layer.

GAT computes attention coefficients between connected nodes, then aggregates neighbor features weighted by those coefficients.

**Signature:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, already includes self-loops
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (F_in, F_out)
- `a` — attention vector (2*F_out,)
- `negative_slope` — LeakyReLU negative slope (default 0.2)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. Project features: `H = X @ W` → (N, F_out)
2. Compute attention scores: `e_ij = LeakyReLU(a[:F_out]·H[i] + a[F_out:]·H[j])`
3. Mask non-edges: set `e[i,j] = -inf` where `A[i,j] = 0`
4. Normalize: `alpha = softmax(e, dim=-1)` over neighbors
5. Aggregate: `output = alpha @ H`

**Hint:** Compute e efficiently as an (N, N) matrix using broadcasting:
`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`

## 中文描述

实现单头图注意力网络（GAT）层。

GAT 计算相连节点之间的注意力系数，然后用这些系数对邻居特征进行加权聚合。

**签名:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值，已包含自环
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (F_in, F_out)
- `a` — 注意力向量 (2*F_out,)
- `negative_slope` — LeakyReLU 负斜率（默认 0.2）

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 特征投影：`H = X @ W` → (N, F_out)
2. 计算注意力分数：`e_ij = LeakyReLU(a[:F_out]·H[i] + a[F_out:]·H[j])`
3. 掩码非边：将 `A[i,j] = 0` 处的 `e[i,j]` 设为 `-inf`
4. 归一化：`alpha = softmax(e, dim=-1)` 对邻居求 softmax
5. 聚合：`output = alpha @ H`

**提示:** 利用广播高效计算 e 矩阵：
`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`

## 提示

```
Compute e as (N,N) via broadcasting: `e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`, then mask with A and softmax.
```

## 参考答案

```python
def gat_layer(A, X, W, a, negative_slope=0.2):
    H = X @ W
    F_out = H.size(-1)
    e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)
    e = torch.nn.functional.leaky_relu(e, negative_slope)
    e = e.masked_fill(A == 0, float('-inf'))
    alpha = torch.softmax(e, dim=-1)
    return alpha @ H
```

