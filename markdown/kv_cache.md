# KV Cache Attention

**中文标题:** KV Cache 注意力

**难度:** Hard

**函数名:** `KVCacheAttention`

## 英文描述

Implement multi-head attention with KV cache for efficient autoregressive decoding.

KV caching stores previously computed keys and values, so each decode step only processes the new token instead of the full sequence.

**Signature:** `KVCacheAttention(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x, cache=None) -> (Tensor, tuple)`
- `x` — input tensor (B, S_new, d_model)
- `cache` — optional (K, V) tuple from previous steps

**Returns:** (output, (K_all, V_all)) where cache grows each step

**Constraints:**
- Concatenate new K/V with cached along sequence dim
- Apply causal mask during prefill (S_new > 1)
- Incremental decode must match full forward numerically

## 中文描述

实现带 KV 缓存的多头注意力，用于高效自回归解码。

KV 缓存存储之前计算的键和值，使每个解码步骤只需处理新 token 而非完整序列。

**签名:** `KVCacheAttention(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x, cache=None) -> (Tensor, tuple)`
- `x` — 输入张量 (B, S_new, d_model)
- `cache` — 可选的前序步骤 (K, V) 元组

**返回:** (输出, (K_all, V_all))，缓存逐步增长

**约束:**
- 将新 K/V 与缓存沿序列维度拼接
- 预填充时（S_new > 1）应用因果掩码
- 增量解码必须与完整前向传播数值一致

## 提示

```
1. Project Q/K/V → reshape to `(B, num_heads, S, d_k)`
2. If cache: K = cat([cache_K, K], dim=2),  V = cat([cache_V, V], dim=2)
3. Apply causal mask when S_new > 1 (prefill)
4. Scaled dot-product attention → output projection
5. Return (output, (K_all, V_all))
```

## 参考答案

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        B, S_new, _ = x.shape

        q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v)
        S_total = k.shape[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if S_new > 1:
            # Causal mask for prefill: each query position can only attend to
            # positions up to itself in the full sequence
            S_past = S_total - S_new
            mask = torch.triu(
                torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
                diagonal=S_past + 1,
            )
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = self.W_o(attn.transpose(1, 2).contiguous().view(B, S_new, -1))
        return out, new_cache
```

## 示例代码

```python
torch.manual_seed(0)
attn = KVCacheAttention(d_model=64, num_heads=4)
x = torch.randn(1, 6, 64)

full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)

print('Full shape:', full_out.shape)
print('Match:', torch.allclose(full_out, inc_out, atol=1e-5))
print('Final cache K shape:', cache[0].shape)
```

