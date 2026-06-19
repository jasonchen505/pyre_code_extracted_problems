# GIN Layer (Graph Isomorphism Network)

**中文标题:** GIN 层（图同构网络）

**难度:** Medium

**函数名:** `gin_layer`

## 英文描述

Implement a Graph Isomorphism Network (GIN) layer.

GIN updates node features by aggregating neighbor features, scaling the node's own features by (1 + eps), and passing through a 2-layer MLP.

**Signature:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F_in)
- `eps` — learnable scalar tensor
- `W1` — first linear weight (F_in, F_hidden)
- `b1` — first linear bias (F_hidden,)
- `W2` — second linear weight (F_hidden, F_out)
- `b2` — second linear bias (F_out,)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. Aggregate neighbor features: `agg = A @ X`
2. Combine: `h = (1 + eps) * X + agg`
3. MLP layer 1: `h = ReLU(h @ W1 + b1)`
4. MLP layer 2: `h = h @ W2 + b2`
5. Return h

## 中文描述

实现图同构网络（GIN）层。

GIN 通过聚合邻居特征、用 (1 + eps) 缩放自身特征，再经过两层 MLP 来更新节点表示。

**签名:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F_in)
- `eps` — 可学习标量张量
- `W1` — 第一层线性权重 (F_in, F_hidden)
- `b1` — 第一层线性偏置 (F_hidden,)
- `W2` — 第二层线性权重 (F_hidden, F_out)
- `b2` — 第二层线性偏置 (F_out,)

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 聚合邻居特征：`agg = A @ X`
2. 组合：`h = (1 + eps) * X + agg`
3. MLP 第一层：`h = ReLU(h @ W1 + b1)`
4. MLP 第二层：`h = h @ W2 + b2`
5. 返回 h

## 提示

```
Aggregate with `A @ X`, combine as `(1 + eps) * X + agg`, then pass through a 2-layer MLP with ReLU.
```

## 参考答案

```python
def gin_layer(A, X, eps, W1, b1, W2, b2):
    agg = A @ X
    h = (1 + eps) * X + agg
    h = torch.relu(h @ W1 + b1)
    return h @ W2 + b2
```

