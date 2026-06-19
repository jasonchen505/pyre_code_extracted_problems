# Cosine LR Scheduler with Warmup

**中文标题:** 余弦学习率调度（含预热）

**难度:** Medium

**函数名:** `cosine_lr_schedule`

## 英文描述

Implement a cosine learning rate schedule with linear warmup.

This scheduler linearly ramps the LR during warmup, then decays it following a cosine curve. Widely used in transformer training.

**Signature:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`

**Parameters:**
- `step` — current training step
- `total_steps` — total number of steps
- `warmup_steps` — number of warmup steps
- `max_lr`, `min_lr` — peak and minimum learning rates

**Returns:** learning rate as a float

**Constraints:**
- Warmup: linear from 0 to max_lr
- Decay: `min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`

## 中文描述

实现带线性预热的余弦学习率调度。

该调度器在预热阶段线性增加学习率，之后按余弦曲线衰减，广泛用于 Transformer 训练。

**签名:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`

**参数:**
- `step` — 当前训练步数
- `total_steps` — 总步数
- `warmup_steps` — 预热步数
- `max_lr`, `min_lr` — 峰值和最小学习率

**返回:** 浮点数学习率

**约束:**
- 预热阶段：从 0 线性增加到 max_lr
- 衰减阶段：`min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`

## 提示

```
1. step < warmup_steps → lr = max_lr * step / warmup_steps
2. step >= total_steps → lr = min_lr
3. progress = (step - warmup_steps) / (total_steps - warmup_steps)
   → min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π·progress))
```

## 参考答案

```python
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

## 示例代码

```python
lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
print(f'Start: {lrs[0]:.6f}, Warmup end: {lrs[10]:.6f}, Mid: {lrs[55]:.6f}, End: {lrs[100]:.6f}')
```

