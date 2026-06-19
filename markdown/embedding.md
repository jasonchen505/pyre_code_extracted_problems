# Embedding Layer

**中文标题:** Embedding 层

**难度:** Easy

**函数名:** `MyEmbedding`

## 英文描述

Implement an embedding lookup layer as an nn.Module.

An embedding layer maps integer indices to dense vectors via a learnable weight matrix, used as the first layer in NLP models.

**Signature:** `MyEmbedding(num_embeddings, embedding_dim)` (nn.Module)

**Forward:** `forward(indices) -> Tensor`
- `indices` — integer tensor of any shape

**Returns:** embedded vectors with an extra trailing dimension of size embedding_dim

**Constraints:**
- Store weights as `nn.Parameter` of shape (num_embeddings, embedding_dim)
- Forward is simply `weight[indices]`

## 中文描述

实现嵌入查找层（nn.Module）。

嵌入层通过可学习的权重矩阵将整数索引映射为稠密向量，是 NLP 模型的第一层。

**签名:** `MyEmbedding(num_embeddings, embedding_dim)`（nn.Module）

**前向传播:** `forward(indices) -> Tensor`
- `indices` — 任意形状的整数张量

**返回:** 嵌入向量，末尾增加 embedding_dim 维度

**约束:**
- 权重存储为 `nn.Parameter`，形状 (num_embeddings, embedding_dim)
- 前向传播即 `weight[indices]`

## 提示

```
`self.weight = nn.Parameter(...)` shape `(num_embeddings, embedding_dim)`
Forward: `return self.weight[indices]`
```

## 参考答案

```python
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]
```

## 示例代码

```python
emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))
```

