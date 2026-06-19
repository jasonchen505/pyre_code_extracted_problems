# Implement RMSNorm

**中文标题:** 实现 RMSNorm

**难度:** Medium

**函数名:** `rms_norm`

## 英文描述

Implement RMSNorm (Root Mean Square Layer Normalization).

RMSNorm is a simpler alternative to LayerNorm that skips mean subtraction, normalizing only by the root mean square of activations.

**Signature:** `rms_norm(x, weight, eps=1e-6) -> Tensor`

**Parameters:**
- `x` — input tensor (..., D)
- `weight` — learnable scale parameter (D,)
- `eps` — epsilon for numerical stability

**Returns:** normalized tensor, same shape as x

**Constraints:**
- `RMS(x) = sqrt(mean(x^2) + eps)` over last dim
- Output: `x / RMS(x) * weight`

## 中文描述

实现 RMSNorm（均方根层归一化）。

RMSNorm 是 LayerNorm 的简化替代，跳过均值减法，仅通过激活值的均方根进行归一化。

**签名:** `rms_norm(x, weight, eps=1e-6) -> Tensor`

**参数:**
- `x` — 输入张量 (..., D)
- `weight` — 可学习的缩放参数 (D,)
- `eps` — 数值稳定性的 epsilon

**返回:** 归一化后的张量，形状与 x 相同

**约束:**
- `RMS(x) = sqrt(mean(x^2) + eps)` 沿最后一维
- 输出：`x / RMS(x) * weight`

## 提示

```
`rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`
`return x / rms * weight`  ← no mean subtraction
```

## 参考答案

```python
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

## 示例代码

```python
x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())
```

