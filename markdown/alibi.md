# ALiBi Attention

**中文标题:** ALiBi 注意力

**难度:** Medium

**函数名:** `alibi_attention`

## 英文描述

Implement Attention with Linear Biases (ALiBi).

ALiBi replaces positional embeddings with a fixed linear bias added to attention scores. Each head gets a different slope `m_h`, penalizing attention to distant tokens.

**Signature:** `alibi_attention(Q, K, V, num_heads) -> Tensor`

**Parameters:**
- `Q, K, V` — tensors of shape `(B, S, D)`
- `num_heads` — number of attention heads

**Returns:** output tensor `(B, S, D)`

**Slope schedule:** `m_h = 1 / 2^(8h/H)` for h = 1..H

**Bias:** `bias[h, i, j] = -m_h * |i - j|` (added to attention scores before softmax)

## 中文描述

实现带线性偏置的注意力（ALiBi）。

ALiBi 用固定线性偏置替代位置嵌入，直接加到注意力分数上。每个头有不同的斜率 `m_h`，对远距离 token 的注意力施加惩罚。

**签名:** `alibi_attention(Q, K, V, num_heads) -> Tensor`

**参数:**
- `Q, K, V` — 形状 `(B, S, D)` 的张量
- `num_heads` — 注意力头数

**返回:** 输出张量 `(B, S, D)`

**斜率:** `m_h = 1 / 2^(8h/H)`，h = 1..H

**偏置:** `bias[h, i, j] = -m_h * |i - j|`（在 softmax 前加到注意力分数上）

## 提示

```
1. slopes: `m_h = 2**(-8*h/H)` for h=1..H
2. distance matrix: `|i-j|`, shape `(S,S)`
3. bias: `-slopes[:, None, None] * distances`, shape `(H,S,S)`
4. standard multi-head attention with bias added to scores
```

## 参考答案

```python
def alibi_attention(Q, K, V, num_heads):
    B, S, D = Q.shape
    d_h = D // num_heads

    # Compute slopes: m_h = 1/2^(8h/H) for h=1..H
    h_idx = torch.arange(1, num_heads + 1, dtype=torch.float32, device=Q.device)
    slopes = 1.0 / (2.0 ** (8.0 * h_idx / num_heads))  # (H,)

    # Distance matrix |i - j|, shape (S, S)
    pos = torch.arange(S, device=Q.device).float()
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S, S)

    # ALiBi bias: (H, S, S)
    bias = -slopes.view(num_heads, 1, 1) * dist.unsqueeze(0)

    # Split into heads: (B, H, S, d_h)
    Qh = Q.view(B, S, num_heads, d_h).transpose(1, 2)
    Kh = K.view(B, S, num_heads, d_h).transpose(1, 2)
    Vh = V.view(B, S, num_heads, d_h).transpose(1, 2)

    scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5) + bias.unsqueeze(0)
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
    return out
```

## 示例代码

```python
torch.manual_seed(0)
B, S, D, H = 2, 6, 16, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)

out = alibi_attention(Q, K, V, num_heads=H)
print("Output shape:", out.shape)

h_idx = torch.arange(1, H + 1, dtype=torch.float32)
slopes = 1.0 / (2.0 ** (8.0 * h_idx / H))
print("Slopes for 4 heads:", slopes)
```

