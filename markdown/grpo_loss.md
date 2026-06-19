# GRPO (Group Relative Policy Optimization) Loss

**中文标题:** GRPO 损失

**难度:** Hard

**函数名:** `grpo_loss`

## 英文描述

Implement the GRPO (Group Relative Policy Optimization) loss.

GRPO normalizes rewards within each prompt group to compute advantages, then optimizes the policy using these group-relative advantages.

**Signature:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`

**Parameters:**
- `logps` — policy log-probabilities (B,)
- `rewards` — scalar rewards (B,)
- `group_ids` — integer group identifiers (B,)
- `eps` — epsilon for numerical stability

**Returns:** scalar loss

**Constraints:**
- Per-group z-score normalization: `A_i = (r_i - mean_g) / (std_g + eps)`
- Advantages must be detached (gradients flow only through logps)
- Loss = `-mean(A_i * logps_i)`

## 中文描述

实现 GRPO（组相对策略优化）损失。

GRPO 在每个提示组内归一化奖励以计算优势值，然后使用这些组相对优势优化策略。

**签名:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`

**参数:**
- `logps` — 策略对数概率 (B,)
- `rewards` — 标量奖励 (B,)
- `group_ids` — 整数组标识符 (B,)
- `eps` — 数值稳定性的 epsilon

**返回:** 标量损失

**约束:**
- 组内 z-score 归一化：`A_i = (r_i - mean_g) / (std_g + eps)`
- 优势值必须 detach（梯度仅通过 logps 流动）
- 损失 = `-mean(A_i * logps_i)`

## 提示

```
1. for each group g: A_i = (r_i - mean_g) / (std_g + eps)
2. advantages = A.detach()  (no grad through rewards)
3. loss = -(advantages * logps).mean()
```

## 参考答案

```python
def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    """Group Relative Policy Optimization (GRPO) loss.

    logps: (B,) policy log-probs for each sampled response
    rewards: (B,) scalar rewards for each response
    group_ids: (B,) integers, same id = same prompt/group
    returns: scalar loss (Tensor)
    """
    # Compute per-group normalized advantages A_i
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)

    # Stop gradient through advantages
    advantages_detached = advantages.detach()

    # GRPO objective: -E[A_i * logpi_i]
    return -(advantages_detached * logps).mean()
```

## 示例代码

```python
logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print('Loss:', grpo_loss(logps, rewards, group_ids).item())
```

