# Causal Self-Attention

**中文标题:** 因果自注意力

**难度:** Medium

**函数名:** `causal_attention`

## 英文描述

Implement causal (masked) self-attention.

Like standard attention but prevents each position from attending to future positions, essential for autoregressive models like GPT.

**Signature:** `causal_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, D)
- `K` — key tensor (B, S, D)
- `V` — value tensor (B, S, D)

**Returns:** causally masked attention output (B, S, D)

**Constraints:**
- Mask future positions with `-inf` before softmax
- Position 0 should only see itself

## 中文描述

实现因果（掩码）自注意力。

与标准注意力类似，但阻止每个位置关注未来位置，这对 GPT 等自回归模型至关重要。

**签名:** `causal_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, D)
- `K` — 键张量 (B, S, D)
- `V` — 值张量 (B, S, D)

**返回:** 因果掩码注意力输出 (B, S, D)

**约束:**
- 在 softmax 之前用 `-inf` 掩盖未来位置
- 位置 0 只能看到自身

## 提示

```
Standard attention + causal mask. `torch.triu(ones, diagonal=1)` → upper triangle → fill with `-inf` before softmax.
```

## 参考答案

```python
def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

## 示例代码

```python
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))
```

