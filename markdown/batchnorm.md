# Implement BatchNorm

**中文标题:** 实现 BatchNorm

**难度:** Medium

**函数名:** `my_batch_norm`

## 英文描述

Implement Batch Normalization with train/eval modes.

BatchNorm normalizes activations per feature using batch statistics during training and running statistics during inference, stabilizing deep network training.

**Signature:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`

**Parameters:**
- `x` — input tensor (N, D)
- `gamma`, `beta` — learnable affine parameters (D,)
- `running_mean`, `running_var` — running statistics (D,), updated in-place during training

**Returns:** normalized and affine-transformed tensor, same shape as x

**Constraints:**
- Training: use batch stats, update running stats with momentum
- Inference: use running stats only
- Use `unbiased=False` for batch variance

## 中文描述

实现带训练/推理模式的批归一化。

BatchNorm 在训练时使用批统计量、推理时使用运行统计量对每个特征进行归一化，从而稳定深度网络训练。

**签名:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`

**参数:**
- `x` — 输入张量 (N, D)
- `gamma`, `beta` — 可学习的仿射参数 (D,)
- `running_mean`, `running_var` — 运行统计量 (D,)，训练时原地更新

**返回:** 归一化并仿射变换后的张量，形状与 x 相同

**约束:**
- 训练模式：使用批统计量，用 momentum 更新运行统计量
- 推理模式：仅使用运行统计量
- 批方差使用 `unbiased=False`

## 提示

```
Train: `mean/var = x.mean/var(dim=0)`, update running stats with `momentum`
Eval: use `running_mean/running_var` directly
Both: `gamma * (x - mean) / sqrt(var + eps) + beta`
```

## 参考答案

```python
import torch

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    """BatchNorm with train/eval behavior and running stats.

    - Training: use batch stats, update running_mean / running_var in-place.
    - Inference: use running_mean / running_var as-is.
    """
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        # Update running statistics in-place. Detach to avoid tracking gradients.
        running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())

        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

## 示例代码

```python
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)

running_mean = torch.zeros(4)
running_var = torch.ones(4)

out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("[Train] Column means:", out_train.mean(dim=0))
print("[Train] Column stds: ", out_train.std(dim=0))
print("Updated running_mean:", running_mean)
print("Updated running_var:", running_var)

out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("[Eval] Column means (using running stats):", out_eval.mean(dim=0))
```

