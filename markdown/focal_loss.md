# Focal Loss

**中文标题:** Focal Loss

**难度:** Medium

**函数名:** `focal_loss`

## 英文描述

Implement Focal Loss for handling class imbalance in classification.

Focal Loss down-weights easy examples so the model focuses on hard ones. It was introduced in RetinaNet for dense object detection.

**Signature:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`

**Parameters:**
- `logits` — raw class scores, shape (N, C)
- `targets` — integer class indices, shape (N,)
- `alpha` — weighting factor (scalar)
- `gamma` — focusing parameter; higher values down-weight easy examples more

**Returns:** scalar mean loss

**Formula:**
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```
where `p_t = softmax(logits)[i, targets[i]]`

**Constraints:**
- Compute softmax manually (no `F.*`)
- Use numerically stable softmax (subtract max before exp)
- Return the mean over the batch

## 中文描述

实现用于处理分类任务中类别不平衡问题的 Focal Loss。

Focal Loss 降低简单样本的权重，使模型专注于困难样本。它由 RetinaNet 在密集目标检测中提出。

**签名:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`

**参数:**
- `logits` — 原始类别分数，形状 (N, C)
- `targets` — 整数类别索引，形状 (N,)
- `alpha` — 加权因子（标量）
- `gamma` — 聚焦参数；值越大，对简单样本的降权越强

**返回:** 标量均值损失

**公式:**
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```
其中 `p_t = softmax(logits)[i, targets[i]]`

**约束:**
- 手动计算 softmax（不得使用 `F.*`）
- 使用数值稳定的 softmax（exp 前减去最大值）
- 返回批次上的均值

## 提示

```
1. Stable softmax: `exp(logits - max)` → normalize → `probs`
2. `p_t = probs[arange(N), targets]`
3. `return (-alpha * (1-p_t)**gamma * log(p_t+1e-8)).mean()`
```

## 参考答案

```python
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    N, C = logits.shape
    # numerically stable softmax
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp_s = torch.exp(shifted)
    probs = exp_s / exp_s.sum(dim=-1, keepdim=True)
    # p_t: probability assigned to the correct class
    p_t = probs[torch.arange(N), targets]
    log_p_t = torch.log(p_t + 1e-8)
    fl = -alpha * (1 - p_t) ** gamma * log_p_t
    return fl.mean()
```

## 示例代码

```python
torch.manual_seed(0)
logits = torch.randn(8, 4)
targets = torch.randint(0, 4, (8,))

fl_gamma0 = focal_loss(logits, targets, alpha=0.25, gamma=0.0)
ce = F.cross_entropy(logits, targets)
print(f"Focal (gamma=0): {fl_gamma0:.4f}  |  alpha * CE: {0.25 * ce:.4f}")

fl_g2 = focal_loss(logits, targets, alpha=0.25, gamma=2.0)
fl_g5 = focal_loss(logits, targets, alpha=0.25, gamma=5.0)
print(f"Focal gamma=2: {fl_g2:.4f}  |  gamma=5: {fl_g5:.4f}  (higher gamma -> lower loss)")
```

