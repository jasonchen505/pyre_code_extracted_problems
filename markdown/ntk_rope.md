# NTK-aware RoPE Scaling

**中文标题:** NTK-aware RoPE 缩放

**难度:** Easy

**函数名:** `ntk_rope`

## 英文描述

Implement NTK-aware RoPE scaling for long-context extrapolation.

Standard RoPE degrades on sequences longer than the training context. NTK-aware scaling adjusts the base frequency so that high-frequency dimensions are preserved while low-frequency ones are stretched, enabling extrapolation without fine-tuning.

**Signature:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`

**Parameters:**
- `q, k` — tensors of shape `(B, S, D)`
- `scale` — context length ratio (new_len / train_len), e.g. 4.0 for 4× context

**Returns:** rotated `(q, k)` with same shapes

**NTK base:** `base_new = 10000 * scale^(D/(D-2))`

Then apply standard RoPE with the new base.

## 中文描述

实现 NTK-aware RoPE 缩放，用于长上下文外推。

标准 RoPE 在超过训练上下文长度的序列上性能下降。NTK-aware 缩放调整基础频率，保留高频维度同时拉伸低频维度，无需微调即可外推。

**签名:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`

**参数:**
- `q, k` — 形状 `(B, S, D)` 的张量
- `scale` — 上下文长度比例（新长度/训练长度），如 4× 上下文传 4.0

**返回:** 旋转后的 `(q, k)`，形状不变

**NTK 基底:** `base_new = 10000 * scale^(D/(D-2))`

然后用新基底应用标准 RoPE。

## 提示

```
`new_base = 10000 * scale**(D/(D-2))` → apply standard RoPE with `new_base` instead of `10000`.
```

## 参考答案

```python
def ntk_rope(q, k, scale):
    B, S, D = q.shape
    new_base = 10000.0 * (scale ** (D / (D - 2)))
    pos = torch.arange(S, device=q.device).float().unsqueeze(1)
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (new_base ** (dim / D))
    angles = pos * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def rotate(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a,
                            x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)

    return rotate(q), rotate(k)
```

## 示例代码

```python
B, S, D = 1, 8, 16
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)

q1, k1 = ntk_rope(q, k, scale=1.0)
q4, k4 = ntk_rope(q, k, scale=4.0)

print("scale=1 base:", 10000.0 * (1.0 ** (D / (D - 2))))
print("scale=4 base:", 10000.0 * (4.0 ** (D / (D - 2))))
print()
print("Norm preservation (scale=1):")
print("  q input norm:", q.norm(dim=-1).mean().item())
print("  q output norm:", q1.norm(dim=-1).mean().item())
print()
print("Norm preservation (scale=4):")
print("  q input norm:", q.norm(dim=-1).mean().item())
print("  q output norm:", q4.norm(dim=-1).mean().item())
```

