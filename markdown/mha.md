# Multi-Head Attention

**中文标题:** 多头注意力

**难度:** Hard

**函数名:** `MultiHeadAttention`

## 英文描述

Implement Multi-Head Attention from scratch.

MHA projects inputs into multiple heads, computes scaled dot-product attention per head, then concatenates and projects the results.

**Signature:** `MultiHeadAttention(d_model, num_heads)`

**Method:** `forward(Q, K, V) -> Tensor`
- `Q` — query tensor (B, S_q, d_model)
- `K` — key tensor (B, S_k, d_model)
- `V` — value tensor (B, S_k, d_model)

**Returns:** attention output (B, S_q, d_model)

**Constraints:**
- Use W_q, W_k, W_v, W_o as `nn.Linear(d_model, d_model)`
- `d_k = d_model // num_heads`
- Support cross-attention (S_q != S_k)

## 中文描述

从零实现多头注意力。

MHA 将输入投影到多个头，每个头计算缩放点积注意力，然后拼接并投影结果。

**签名:** `MultiHeadAttention(d_model, num_heads)`

**方法:** `forward(Q, K, V) -> Tensor`
- `Q` — 查询张量 (B, S_q, d_model)
- `K` — 键张量 (B, S_k, d_model)
- `V` — 值张量 (B, S_k, d_model)

**返回:** 注意力输出 (B, S_q, d_model)

**约束:**
- 使用 W_q、W_k、W_v、W_o 作为 `nn.Linear(d_model, d_model)`
- `d_k = d_model // num_heads`
- 支持交叉注意力（S_q != S_k）

## 提示

```
1. Project Q/K/V with `nn.Linear(d_model, d_model)`
2. Reshape to `(B, heads, S, d_k)` via `.view(...).transpose(1,2)`
3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`
4. Transpose + reshape → `W_o` projection
```

## 参考答案

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(out)
```

## 示例代码

```python
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Self-attn shape:", out.shape)

Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)
```

