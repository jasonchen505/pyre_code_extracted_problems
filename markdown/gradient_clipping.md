# Gradient Norm Clipping

**中文标题:** 梯度范数裁剪

**难度:** Easy

**函数名:** `clip_grad_norm`

## 英文描述

Implement gradient norm clipping.

Gradient clipping rescales all parameter gradients when their combined L2 norm exceeds a threshold, preventing exploding gradients.

**Signature:** `clip_grad_norm(parameters, max_norm) -> float`

**Parameters:**
- `parameters` — list of tensors with `.grad` attributes
- `max_norm` — maximum allowed gradient norm

**Returns:** original total gradient norm (float)

**Constraints:**
- Total norm = `sqrt(sum(p.grad.norm()^2))`
- Only clip if total norm > max_norm
- Preserve gradient direction

## 中文描述

实现梯度范数裁剪。

梯度裁剪在所有参数梯度的 L2 范数超过阈值时进行缩放，防止梯度爆炸。

**签名:** `clip_grad_norm(parameters, max_norm) -> float`

**参数:**
- `parameters` — 带 `.grad` 属性的张量列表
- `max_norm` — 允许的最大梯度范数

**返回:** 原始总梯度范数（浮点数）

**约束:**
- 总范数 = `sqrt(sum(p.grad.norm()^2))`
- 仅在总范数 > max_norm 时裁剪
- 保持梯度方向不变

## 提示

```
1. total_norm = sqrt(Σ p.grad.norm()²)
2. clip_coef = max_norm / total_norm
3. if total_norm > max_norm: scale all grads by clip_coef
4. return original total_norm (float)
```

## 参考答案

```python
def clip_grad_norm(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm.item()
```

## 示例代码

```python
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Returned:', orig)
```

