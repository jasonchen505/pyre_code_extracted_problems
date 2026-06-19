# Multi-Head Cross-Attention

**中文标题:** 多头交叉注意力

**难度:** Medium

**函数名:** `MultiHeadCrossAttention`

## 英文描述

Implement multi-head cross-attention as an nn.Module.

Cross-attention lets a decoder attend to encoder outputs: Q comes from one sequence, K/V from another. No causal mask is applied.

**Signature:** `MultiHeadCrossAttention(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x_q, x_kv) -> Tensor`
- `x_q` — query input (B, S_q, d_model)
- `x_kv` — key/value input (B, S_kv, d_model)

**Returns:** attention output (B, S_q, d_model)

**Constraints:**
- Use separate W_q, W_k, W_v, W_o linear projections
- Q and KV can have different sequence lengths

## 中文描述

实现多头交叉注意力（nn.Module）。

交叉注意力让解码器关注编码器输出：Q 来自一个序列，K/V 来自另一个序列，不使用因果掩码。

**签名:** `MultiHeadCrossAttention(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x_q, x_kv) -> Tensor`
- `x_q` — 查询输入 (B, S_q, d_model)
- `x_kv` — 键/值输入 (B, S_kv, d_model)

**返回:** 注意力输出 (B, S_q, d_model)

**约束:**
- 使用独立的 W_q、W_k、W_v、W_o 线性投影
- Q 和 KV 可以有不同的序列长度

## 提示

```
Q from `x_q`, K/V from `x_kv`. Project → reshape to `(B, H, S, d_k)` → scaled dot-product (no causal mask) → concat heads → `W_o`.
```

## 参考答案

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape
        S_kv = x_kv.shape[1]
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))
```

## 示例代码

```python
attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)
x_kv = torch.randn(2, 10, 64)
print('Output:', attn(x_q, x_kv).shape)
```

