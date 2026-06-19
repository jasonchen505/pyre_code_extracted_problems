# Cross-Entropy Loss

**中文标题:** 交叉熵损失

**难度:** Easy

**函数名:** `cross_entropy_loss`

## 英文描述

Implement cross-entropy loss for classification.

Cross-entropy measures the difference between predicted logits and true class labels. It is the standard loss for classification tasks.

**Signature:** `cross_entropy_loss(logits, targets) -> Tensor`

**Parameters:**
- `logits` — raw scores (B, C) where C is the number of classes
- `targets` — ground-truth class indices (B,)

**Returns:** scalar mean loss

**Constraints:**
- Must be numerically stable (handle large logits)
- Use log-sum-exp trick for stability

## 中文描述

实现分类交叉熵损失。

交叉熵衡量预测 logits 与真实类别标签之间的差异，是分类任务的标准损失函数。

**签名:** `cross_entropy_loss(logits, targets) -> Tensor`

**参数:**
- `logits` — 原始分数 (B, C)，C 为类别数
- `targets` — 真实类别索引 (B,)

**返回:** 标量平均损失

**约束:**
- 必须数值稳定（处理大 logits）
- 使用 log-sum-exp 技巧保证稳定性

## 提示

```
1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`
2. `return -log_probs[arange(B), targets].mean()`
```

## 参考答案

```python
def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()
```

## 示例代码

```python
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())
```

