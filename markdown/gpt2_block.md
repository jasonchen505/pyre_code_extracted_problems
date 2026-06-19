# GPT-2 Transformer Block

**中文标题:** GPT-2 Transformer Block

**难度:** Hard

**函数名:** `GPT2Block`

## 英文描述

Implement a GPT-2 transformer block as an nn.Module.

A GPT-2 block uses pre-norm architecture: LayerNorm before causal self-attention and MLP, with residual connections around both.

**Signature:** `GPT2Block(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** output tensor (B, S, d_model)

**Constraints:**
- Pre-norm: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`
- MLP: Linear(d, 4d) -> GELU -> Linear(4d, d)
- Attention must be causal (future tokens cannot affect past)

## 中文描述

实现 GPT-2 Transformer 块（nn.Module）。

GPT-2 块使用 pre-norm 架构：在因果自注意力和 MLP 之前进行 LayerNorm，两者都有残差连接。

**签名:** `GPT2Block(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 输出张量 (B, S, d_model)

**约束:**
- Pre-norm：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`
- MLP：Linear(d, 4d) -> GELU -> Linear(4d, d)
- 注意力必须是因果的（未来 token 不能影响过去）

## 提示

```
Pre-norm residual: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`. MLP: `Linear(d,4d) → GELU → Linear(4d,d)`. Attention must be causal (mask future with `-inf`).
```

## 参考答案

```python
class _GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            _GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

## 示例代码

```python
block = GPT2Block(64, 4)
print('Output:', block(torch.randn(2, 8, 64)).shape)
print('Params:', sum(p.numel() for p in block.parameters()))
```

