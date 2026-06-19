# Graph Readout (Graph-Level Pooling)

**中文标题:** 图读出（图级池化）

**难度:** Easy

**函数名:** `graph_readout`

## 英文描述

Implement graph-level readout (pooling) that aggregates node embeddings into a single vector per graph.

**Signature:** `graph_readout(X, batch, mode='mean') -> Tensor`

**Parameters:**
- `X` — node embedding matrix (total_nodes, D)
- `batch` — integer tensor (total_nodes,) mapping each node to its graph index (0-indexed)
- `mode` — aggregation mode: `'mean'`, `'sum'`, or `'max'`

**Returns:** graph-level embeddings of shape (num_graphs, D) where `num_graphs = batch.max() + 1`

**Algorithm:**
- `'sum'`: accumulate node features per graph (e.g., `index_add_` or `scatter_add_`)
- `'mean'`: sum per graph divided by node count per graph
- `'max'`: per-graph element-wise maximum over node features

**Constraints:**
- Do not use PyG or DGL — use only raw PyTorch operations
- Handle all three modes

## 中文描述

实现图级别的读出（池化）操作，将节点嵌入聚合为每个图的单一向量表示。

**签名:** `graph_readout(X, batch, mode='mean') -> Tensor`

**参数:**
- `X` — 节点嵌入矩阵 (total_nodes, D)
- `batch` — 整数张量 (total_nodes,)，将每个节点映射到所属图的索引（从 0 开始）
- `mode` — 聚合方式：`'mean'`、`'sum'` 或 `'max'`

**返回:** 形状为 (num_graphs, D) 的图级嵌入，其中 `num_graphs = batch.max() + 1`

**算法:**
- `'sum'`：按图累加节点特征（可用 `index_add_` 或 `scatter_add_`）
- `'mean'`：按图求和后除以各图节点数
- `'max'`：按图对节点特征逐元素取最大值

**约束:**
- 仅使用原生 PyTorch 操作，不得使用 PyG 或 DGL
- 需支持全部三种聚合模式

## 提示

```
1. `num_graphs = batch.max().item() + 1`
2. For sum: `out.index_add_(0, batch, X)`
3. For mean: sum / count per graph
4. For max: loop over graph indices or use `scatter` with `reduce='amax'`
```

## 参考答案

```python
def graph_readout(X, batch, mode='mean'):
    import torch
    num_graphs = batch.max().item() + 1
    D = X.shape[1]
    if mode == 'sum':
        out = torch.zeros(num_graphs, D)
        out.index_add_(0, batch, X)
        return out
    elif mode == 'mean':
        out = torch.zeros(num_graphs, D)
        out.index_add_(0, batch, X)
        count = torch.zeros(num_graphs)
        count.index_add_(0, batch, torch.ones(batch.shape[0]))
        return out / count.unsqueeze(1)
    elif mode == 'max':
        out = torch.full((num_graphs, D), float('-inf'))
        for i in range(num_graphs):
            mask = batch == i
            out[i] = X[mask].max(dim=0).values
        return out
    else:
        raise ValueError(f"Unsupported mode: {mode}")
```

