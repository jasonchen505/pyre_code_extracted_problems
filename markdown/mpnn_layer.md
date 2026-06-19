# MPNN Layer (Message Passing)

**中文标题:** MPNN 层（消息传递）

**难度:** Medium

**函数名:** `mpnn_layer`

## 英文描述

Implement a Message Passing Neural Network (MPNN) layer.

MPNN computes messages along edges using node and edge features, aggregates them per node, then updates node representations.

**Signature:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F)
- `E` — edge feature tensor (N, N, D_e)
- `W_msg` — message weight matrix (2*F + D_e, F_msg)
- `W_upd` — update weight matrix (F + F_msg, F_out)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. For each edge (i,j) where A[i,j]=1, compute message: `m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`
2. For each node i, aggregate: `agg_i = sum of m_ij over neighbors j`
3. For each node i, update: `out_i = ReLU(concat(X[i], agg_i) @ W_upd)`

**Hint:** Build an (N, N, 2F+D_e) tensor by broadcasting X, then matmul with W_msg, mask by A, and sum over dim=1.

## 中文描述

实现消息传递神经网络（MPNN）层。

MPNN 利用节点和边特征沿边计算消息，按节点聚合后更新节点表示。

**签名:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F)
- `E` — 边特征张量 (N, N, D_e)
- `W_msg` — 消息权重矩阵 (2*F + D_e, F_msg)
- `W_upd` — 更新权重矩阵 (F + F_msg, F_out)

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 对每条边 (i,j)（A[i,j]=1），计算消息：`m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`
2. 对每个节点 i，聚合：`agg_i = 对邻居 j 的 m_ij 求和`
3. 对每个节点 i，更新：`out_i = ReLU(concat(X[i], agg_i) @ W_upd)`

**提示:** 利用广播构建 (N, N, 2F+D_e) 张量，与 W_msg 矩阵乘，用 A 掩码后沿 dim=1 求和。

## 提示

```
Build (N, N, 2F+D_e) by broadcasting `X[i]` and `X[j]` with `E[i,j]`, matmul with W_msg, mask by A, sum over dim=1.
```

## 参考答案

```python
def mpnn_layer(A, X, E, W_msg, W_upd):
    N, F = X.shape
    Xi = X.unsqueeze(1).expand(N, N, F)
    Xj = X.unsqueeze(0).expand(N, N, F)
    inp = torch.cat([Xi, Xj, E], dim=-1)
    msgs = torch.relu(inp @ W_msg)
    msgs = msgs * A.unsqueeze(-1)
    agg = msgs.sum(dim=1)
    return torch.relu(torch.cat([X, agg], dim=-1) @ W_upd)
```

