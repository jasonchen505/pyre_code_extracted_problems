# PPO (Proximal Policy Optimization) Clipped Loss

**中文标题:** PPO 损失

**难度:** Medium

**函数名:** `ppo_loss`

## 英文描述

Implement the PPO clipped surrogate loss.

PPO constrains policy updates by clipping the importance sampling ratio, preventing destructively large updates during reinforcement learning.

**Signature:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`

**Parameters:**
- `new_logps` — current policy log-probs (B,)
- `old_logps` — old policy log-probs (B,), treated as constant
- `advantages` — advantage estimates (B,), treated as constant
- `clip_ratio` — clipping range epsilon

**Returns:** scalar loss

**Constraints:**
- Ratio: `r = exp(new_logps - old_logps.detach())`
- Loss: `-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`
- Gradients flow only through new_logps

## 中文描述

实现 PPO 裁剪代理损失。

PPO 通过裁剪重要性采样比率来约束策略更新，防止强化学习中的破坏性大幅更新。

**签名:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`

**参数:**
- `new_logps` — 当前策略对数概率 (B,)
- `old_logps` — 旧策略对数概率 (B,)，视为常量
- `advantages` — 优势估计 (B,)，视为常量
- `clip_ratio` — 裁剪范围 epsilon

**返回:** 标量损失

**约束:**
- 比率：`r = exp(new_logps - old_logps.detach())`
- 损失：`-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`
- 梯度仅通过 new_logps 流动

## 提示

```
1. r = exp(new_logps - old_logps.detach())
2. unclipped = r * advantages.detach()
3. clipped = clamp(r, 1-ε, 1+ε) * advantages.detach()
4. loss = -min(unclipped, clipped).mean()
```

## 参考答案

```python
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    """PPO clipped surrogate loss.

    new_logps: (B,) current policy log-probs
    old_logps: (B,) old policy log-probs (treated as constant)
    advantages: (B,) advantage estimates (treated as constant)
    returns: scalar loss (Tensor)
    """
    # Detach old_logps and advantages so gradients only flow through new_logps
    old_logps_detached = old_logps.detach()
    adv_detached = advantages.detach()

    # Importance sampling ratio r = pi_new / pi_old in log-space
    ratios = torch.exp(new_logps - old_logps_detached)

    # Unclipped and clipped objectives
    unclipped = ratios * adv_detached
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached

    # PPO objective: negative mean of the more conservative objective
    return -torch.min(unclipped, clipped).mean()
```

## 示例代码

```python
new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
print('Loss:', ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2))
```

