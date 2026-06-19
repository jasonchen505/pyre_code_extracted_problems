# Rotary Position Embedding (RoPE)

**中文标题:** 旋转位置编码（RoPE）

**难度:** Medium

**函数名:** `apply_rope`

## 英文描述

Implement Rotary Position Embedding (RoPE).

RoPE encodes position information by rotating query and key vectors in pairs, enabling relative position awareness through dot-product properties.

**Signature:** `apply_rope(q, k) -> (Tensor, Tensor)`

**Parameters:**
- `q` — query tensor (B, S, D)
- `k` — key tensor (B, S, D)

**Returns:** tuple of rotated (q, k) with same shapes

**Constraints:**
- Split into even/odd pairs, apply rotation with `angles = pos * 1/(10000^(2i/D))`
- Must preserve vector norms
- Dot products should depend only on relative position

## 中文描述

实现旋转位置编码（RoPE）。

RoPE 通过成对旋转查询和键向量来编码位置信息，利用点积性质实现相对位置感知。

**签名:** `apply_rope(q, k) -> (Tensor, Tensor)`

**参数:**
- `q` — 查询张量 (B, S, D)
- `k` — 键张量 (B, S, D)

**返回:** 旋转后的 (q, k) 元组，形状不变

**约束:**
- 按奇偶对分割，使用 `angles = pos * 1/(10000^(2i/D))` 旋转
- 必须保持向量范数不变
- 点积应仅依赖于相对位置

## 提示

```
1. `freqs = 1 / 10000^(2i/D)` for i in `range(D//2)`
2. `angles = pos[:, None] * freqs` → `cos_a`, `sin_a`
3. Split: `x_e = x[..., 0::2]`, `x_o = x[..., 1::2]`
4. Rotate: `[x_e*cos - x_o*sin, x_e*sin + x_o*cos]` → stack → flatten
```

## 参考答案

```python
def apply_rope(q, k):
    B, S, D = q.shape
    pos = torch.arange(S, device=q.device).unsqueeze(1).float()
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (10000.0 ** (dim / D))
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
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))
```

