# SwiGLU MLP

**中文标题:** SwiGLU MLP

**难度:** Medium

**函数名:** `SwiGLUMLP`

## 英文描述

Implement a SwiGLU MLP as an nn.Module.

SwiGLU is a gated MLP variant used in LLaMA and other modern LLMs: it applies SiLU gating to one projection and element-wise multiplies with another.

**Signature:** `SwiGLUMLP(d_model, d_ff)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, d_model)

**Returns:** output tensor (*, d_model)

**Constraints:**
- Three projections: gate_proj(d, d_ff), up_proj(d, d_ff), down_proj(d_ff, d)
- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`
- SiLU(x) = x * sigmoid(x)

## 中文描述

实现 SwiGLU MLP（nn.Module）。

SwiGLU 是 LLaMA 等现代 LLM 使用的门控 MLP 变体：对一个投影应用 SiLU 门控，与另一个投影逐元素相乘。

**签名:** `SwiGLUMLP(d_model, d_ff)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, d_model)

**返回:** 输出张量 (*, d_model)

**约束:**
- 三个投影：gate_proj(d, d_ff)、up_proj(d, d_ff)、down_proj(d_ff, d)
- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`
- SiLU(x) = x * sigmoid(x)

## 提示

```
`gate_proj(d→d_ff)`, `up_proj(d→d_ff)`, `down_proj(d_ff→d)`. Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`. `silu(x) = x * sigmoid(x)`.
```

## 参考答案

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate = self.gate_proj(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.down_proj(silu_gate * self.up_proj(x))
```

## 示例代码

```python
mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print('Output:', mlp(x).shape)
print('Params:', sum(p.numel() for p in mlp.parameters()))
```

