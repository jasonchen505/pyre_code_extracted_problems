# Label Smoothing Loss

**中文标题:** 标签平滑损失

**难度:** Easy

**函数名:** `label_smoothing`

## 英文描述

Implement label smoothing cross-entropy loss.

Label smoothing prevents overconfidence by mixing the one-hot target distribution with a uniform distribution. It is widely used in Transformer training.

**Signature:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`

**Parameters:**
- `logits` — raw model outputs, shape `(N, C)`
- `targets` — integer class indices, shape `(N,)`
- `smoothing` — smoothing factor ε ∈ [0, 1)

**Returns:** scalar loss

**Formula:** soft target for correct class = `1 - ε`, for others = `ε / (C - 1)`

## 中文描述

实现标签平滑交叉熵损失。

标签平滑通过将 one-hot 目标分布与均匀分布混合来防止过度自信，在 Transformer 训练中被广泛使用。

**签名:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`

**参数:**
- `logits` — 模型原始输出，形状 `(N, C)`
- `targets` — 整数类别索引，形状 `(N,)`
- `smoothing` — 平滑系数 ε ∈ [0, 1)

**返回:** 标量损失

**公式:** 正确类别的软目标 = `1 - ε`，其他类别 = `ε / (C - 1)`

## 提示

```
1. `soft = full((N,C), ε/(C-1))`, then `soft.scatter_(1, targets, 1-ε)`
2. `log_probs` via stable log-softmax
3. `return -(soft * log_probs).sum(dim=-1).mean()`
```

## 参考答案

```python
def label_smoothing(logits, targets, smoothing=0.1):
    N, C = logits.shape
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    soft_targets = torch.full_like(log_probs, smoothing / (C - 1))
    soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

## 示例代码

```python
torch.manual_seed(0)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))

loss_smoothed = label_smoothing(logits, targets, smoothing=0.1)
loss_ce = torch.nn.functional.cross_entropy(logits, targets)

print(f"Label smoothing loss (eps=0.1): {loss_smoothed.item():.4f}")
print(f"Standard CE loss (eps=0.0):    {loss_ce.item():.4f}")

loss_no_smooth = label_smoothing(logits, targets, smoothing=0.0)
print(f"Label smoothing loss (eps=0.0): {loss_no_smooth.item():.4f}  (matches CE: {torch.allclose(loss_no_smooth, loss_ce)})")
```

