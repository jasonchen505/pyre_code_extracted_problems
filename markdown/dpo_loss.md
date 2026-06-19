# DPO (Direct Preference Optimization) Loss

**中文标题:** DPO 损失

**难度:** Medium

**函数名:** `dpo_loss`

## 英文描述

Implement the DPO (Direct Preference Optimization) loss.

DPO aligns language models with human preferences without reinforcement learning, using paired chosen/rejected log-probabilities.

**Signature:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`

**Parameters:**
- `policy_chosen_logps` — policy log-probs for chosen responses (B,)
- `policy_rejected_logps` — policy log-probs for rejected responses (B,)
- `ref_chosen_logps`, `ref_rejected_logps` — reference model log-probs (B,)
- `beta` — temperature scaling factor

**Returns:** scalar loss

**Constraints:**
- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`

## 中文描述

实现 DPO（直接偏好优化）损失。

DPO 无需强化学习即可将语言模型与人类偏好对齐，使用配对的选中/拒绝对数概率。

**签名:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`

**参数:**
- `policy_chosen_logps` — 策略模型对选中回复的对数概率 (B,)
- `policy_rejected_logps` — 策略模型对拒绝回复的对数概率 (B,)
- `ref_chosen_logps`, `ref_rejected_logps` — 参考模型的对数概率 (B,)
- `beta` — 温度缩放因子

**返回:** 标量损失

**约束:**
- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`

## 提示

```
1. chosen_logratios = β·(π_chosen - ref_chosen)
2. rejected_logratios = β·(π_rejected - ref_rejected)
3. loss = -log(sigmoid(chosen_logratios - rejected_logratios)).mean()
```

## 参考答案

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    diff = chosen_rewards - rejected_rewards
    return -torch.log(torch.sigmoid(diff)).mean()
```

## 示例代码

```python
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
```

