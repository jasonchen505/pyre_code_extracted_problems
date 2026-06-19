# Softmax Attention

**中文标题:** Softmax 注意力

**难度:** Easy

**函数名:** `scaled_dot_product_attention`

## 英文描述

Implement scaled dot-product attention.

This is the core attention mechanism: compute similarity scores between queries and keys, then use them to weight values.

**Signature:** `scaled_dot_product_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S_q, D)
- `K` — key tensor (B, S_k, D)
- `V` — value tensor (B, S_k, D_v)

**Returns:** weighted values tensor (B, S_q, D_v)

**Constraints:**
- Scale scores by `1/sqrt(d_k)`
- Support cross-attention (S_q != S_k)

## 中文描述

实现缩放点积注意力。

这是核心注意力机制：计算查询和键之间的相似度分数，然后用它们对值进行加权。

**签名:** `scaled_dot_product_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S_q, D)
- `K` — 键张量 (B, S_k, D)
- `V` — 值张量 (B, S_k, D_v)

**返回:** 加权后的值张量 (B, S_q, D_v)

**约束:**
- 分数需除以 `sqrt(d_k)` 进行缩放
- 支持交叉注意力（S_q != S_k）

## 提示

```
`scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)` → `torch.bmm(softmax(scores, dim=-1), V)`.
```

## 参考答案

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

## 示例代码

```python
torch.manual_seed(42)
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

out = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)
print("Attention weights sum to 1?", True)

Q2 = torch.randn(1, 3, 16)
K2 = torch.randn(1, 5, 16)
V2 = torch.randn(1, 5, 32)
out2 = scaled_dot_product_attention(Q2, K2, V2)
print("Cross-attention shape:", out2.shape, "(expected: 1, 3, 32)")
```

