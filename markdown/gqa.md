# Grouped Query Attention

**中文标题:** 分组查询注意力（GQA）

**难度:** Hard

**函数名:** `GroupQueryAttention`

## 英文描述

Implement Grouped Query Attention (GQA).

GQA uses fewer KV heads than query heads, sharing each KV head across a group of query heads. This reduces KV cache size while preserving quality.

**Signature:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** attention output (B, S, d_model)

**Constraints:**
- W_k/W_v project to `num_kv_heads * d_k` dimensions
- Expand KV heads with `repeat_interleave` to match query heads
- Degenerates to standard MHA when `num_kv_heads == num_heads`

## 中文描述

实现分组查询注意力（GQA）。

GQA 使用比查询头更少的 KV 头，每个 KV 头在一组查询头之间共享，在保持质量的同时减少 KV 缓存大小。

**签名:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 注意力输出 (B, S, d_model)

**约束:**
- W_k/W_v 投影到 `num_kv_heads * d_k` 维
- 使用 `repeat_interleave` 扩展 KV 头以匹配查询头数
- 当 `num_kv_heads == num_heads` 时退化为标准 MHA

## 提示

```
1. `W_q` → `(B,H,S,d_k)`, `W_k`/`W_v` → `(B,KV,S,d_k)`
2. `K = K.repeat_interleave(H//KV, dim=1)` to expand KV heads
3. Scaled dot-product attn → reshape to `(B,S,d_model)`
```

## 参考答案

```python
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

## 示例代码

```python
gqa = GroupQueryAttention(32, 8, 2)
print('Output:', gqa.forward(torch.randn(1, 4, 32)).shape)
```

