# ViT Transformer Block

**中文标题:** ViT Transformer Block

**难度:** Hard

**函数名:** `ViTBlock`

## 英文描述

Implement a Vision Transformer (ViT) block as an nn.Module.

A ViT block uses pre-norm architecture with multi-head self-attention and an MLP, both wrapped in residual connections.

**Signature:** `ViTBlock(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, N, d_model), where N = num_patches + 1 (includes CLS token)

**Returns:** output tensor (B, N, d_model)

**Architecture:**
```
x = x + MHA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Constraints:**
- `nn.Linear` and `nn.LayerNorm` are allowed as building blocks
- MHA must be hand-implemented (no `nn.MultiheadAttention`)
- MLP: `Linear(d, 4d) -> GELU -> Linear(4d, d)`
- GELU must be hand-implemented: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- No `F.*` or `nn.functional.*` calls anywhere

## 中文描述

将 Vision Transformer（ViT）块实现为 nn.Module。

ViT 块使用 pre-norm 架构，包含多头自注意力和 MLP，两者都有残差连接。

**签名:** `ViTBlock(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, N, d_model)，其中 N = 图像块数 + 1（含 CLS token）

**返回:** 输出张量 (B, N, d_model)

**架构:**
```
x = x + MHA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**约束:**
- `nn.Linear` 和 `nn.LayerNorm` 可作为构建块使用
- MHA 必须手动实现（不得使用 `nn.MultiheadAttention`）
- MLP：`Linear(d, 4d) -> GELU -> Linear(4d, d)`
- GELU 必须手动实现：`x * 0.5 * (1 + erf(x / sqrt(2)))`
- 任何地方都不得调用 `F.*` 或 `nn.functional.*`

## 提示

```
MHA: `nn.Linear(d, 3d)` → split Q/K/V → reshape `(B, H, N, d_h)` → scaled dot-product → `proj`
GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`
Block: `x = x + MHA(norm1(x))` → `x = x + MLP(norm2(x))`
```

## 参考答案

```python
import math as _math

class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / _math.sqrt(2.0)))

    def _mha(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_h ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def forward(self, x):
        x = x + self._mha(self.norm1(x))
        x = x + self.fc2(self._gelu(self.fc1(self.norm2(x))))
        return x
```

## 示例代码

```python
torch.manual_seed(0)
batch, num_patches, d_model, num_heads = 2, 16, 64, 4

block = ViTBlock(d_model, num_heads)
x = torch.randn(batch, num_patches, d_model)
out = block(x)

print("Input shape: ", x.shape)    # (2, 16, 64)
print("Output shape:", out.shape)  # (2, 16, 64)
assert out.shape == x.shape, "Shape mismatch!"
print("Shape preserved: True")
```

