# Bradley-Terry Reward Model Loss

**中文标题:** Bradley-Terry 奖励模型

**难度:** Medium

**函数名:** `reward_model_loss`

## 英文描述

Implement the canonical RLHF reward model training loss based on the Bradley-Terry preference model.

Given hidden states for chosen and rejected responses, project them to scalar rewards and compute the preference loss.

**Signature:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`

**Parameters:**
- `chosen_hidden` — last-token hidden states for chosen responses (B, D)
- `rejected_hidden` — last-token hidden states for rejected responses (B, D)
- `reward_head` — linear projection to scalar reward (D, 1)

**Returns:** scalar loss tensor

**Formula:**
1. `r_chosen = chosen_hidden @ reward_head` — shape (B, 1)
2. `r_rejected = rejected_hidden @ reward_head` — shape (B, 1)
3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`

**Constraint:** implement sigmoid manually — `σ(x) = 1 / (1 + exp(-x))` — do not use library sigmoid functions.

## 中文描述

实现基于 Bradley-Terry 偏好模型的标准 RLHF 奖励模型训练损失。

给定选中和拒绝响应的隐藏状态，将其投影为标量奖励并计算偏好损失。

**签名:** `reward_model_loss(chosen_hidden, rejected_hidden, reward_head) -> Tensor`

**参数:**
- `chosen_hidden` — 选中响应的最后一个 token 隐藏状态 (B, D)
- `rejected_hidden` — 拒绝响应的最后一个 token 隐藏状态 (B, D)
- `reward_head` — 投影到标量奖励的线性层权重 (D, 1)

**返回:** 标量损失张量

**公式:**
1. `r_chosen = chosen_hidden @ reward_head` — 形状 (B, 1)
2. `r_rejected = rejected_hidden @ reward_head` — 形状 (B, 1)
3. `loss = -mean(log(sigmoid(r_chosen - r_rejected)))`

**约束:** 手动实现 sigmoid — `σ(x) = 1 / (1 + exp(-x))` — 不得使用库函数。

## 提示

```
1. r_chosen = (chosen_hidden @ reward_head).squeeze(-1)  → (B,)
2. r_rejected = (rejected_hidden @ reward_head).squeeze(-1)
3. margin = r_chosen - r_rejected
4. loss = -log(1 / (1 + exp(-margin))).mean()  (manual sigmoid)
```

## 参考答案

```python
def reward_model_loss(chosen_hidden, rejected_hidden, reward_head):
    r_chosen = (chosen_hidden @ reward_head).squeeze(-1)     # (B,)
    r_rejected = (rejected_hidden @ reward_head).squeeze(-1) # (B,)
    margin = r_chosen - r_rejected
    # manual log-sigmoid: log(1/(1+exp(-x))) = -log(1+exp(-x))
    loss = -torch.log(1.0 / (1.0 + torch.exp(-margin))).mean()
    return loss
```

## 示例代码

```python
torch.manual_seed(0)

B, D = 8, 64
reward_head = torch.randn(D, 1)

h = torch.randn(B, D)
loss_equal = reward_model_loss(h, h, reward_head)
print(f"Equal hidden states => loss = {loss_equal.item():.4f}  (expected ≈ {math.log(2):.4f})")

chosen  = torch.ones(B, D)
rejected = -torch.ones(B, D)
loss_good = reward_model_loss(chosen, rejected, reward_head)
print(f"Chosen >> rejected  => loss = {loss_good.item():.4f}  (expected small)")
```

