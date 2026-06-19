# MoE Load Balancing Loss

**中文标题:** MoE 负载均衡损失

**难度:** Medium

**函数名:** `moe_load_balance_loss`

## 英文描述

Implement the auxiliary load balancing loss from Switch Transformer / DeepSeek-style Mixture-of-Experts.

Without this loss, MoE models collapse: all tokens route to the same expert, wasting capacity. The loss penalizes uneven token distribution by combining a non-differentiable routing fraction with a differentiable probability term.

**Signature:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`

**Parameters:**
- `router_logits` — raw router scores before softmax (N_tokens, num_experts)
- `num_experts` — number of experts

**Returns:** scalar auxiliary loss

**Formula:**
- `f_i` = fraction of tokens assigned to expert i (via argmax — non-differentiable)
- `P_i` = mean router probability for expert i (via softmax — differentiable)
- `L_aux = num_experts * Σ(f_i * P_i)`

**Note:** Gradient flows only through `P_i` (the softmax term). `f_i` is a stop-gradient counting term.

## 中文描述

实现 Switch Transformer / DeepSeek 风格混合专家（MoE）中的辅助负载均衡损失。

没有这个损失，MoE 模型会退化：所有 token 都路由到同一个专家，浪费容量。该损失通过将不可微的路由分数与可微的概率项结合，惩罚不均匀的 token 分布。

**签名:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`

**参数:**
- `router_logits` — softmax 前的原始路由分数 (N_tokens, num_experts)
- `num_experts` — 专家数量

**返回:** 标量辅助损失

**公式:**
- `f_i` = 路由到专家 i 的 token 比例（通过 argmax，不可微）
- `P_i` = 专家 i 的平均路由概率（通过 softmax，可微）
- `L_aux = num_experts * Σ(f_i * P_i)`

**注意:** 梯度只通过 `P_i`（softmax 项）传播，`f_i` 是停止梯度的计数项。

## 提示

```
1. `assignments = logits.argmax(dim=-1)` → `f[e] = (assignments == e).float().mean()`
2. `P = softmax(logits, dim=-1).mean(dim=0)`
3. return `num_experts * (f * P).sum()`
```

## 参考答案

```python
def moe_load_balance_loss(router_logits, num_experts):
    N = router_logits.shape[0]
    # f_i: fraction of tokens routed to each expert (argmax, non-differentiable)
    assignments = router_logits.argmax(dim=-1)  # (N,)
    f = torch.zeros(num_experts, device=router_logits.device)
    for e in range(num_experts):
        f[e] = (assignments == e).float().mean()
    # P_i: mean router probability per expert (differentiable via softmax)
    probs = torch.softmax(router_logits, dim=-1)  # (N, num_experts)
    P = probs.mean(dim=0)                          # (num_experts,)
    return num_experts * (f * P).sum()
```

## 示例代码

```python
num_experts = 4
N_tokens = 100
logits_uniform = torch.zeros(N_tokens, num_experts)
loss_uniform = moe_load_balance_loss(logits_uniform, num_experts)
print(f"Uniform routing loss: {loss_uniform.item():.4f}  (expected 1.0)")

logits_collapsed = torch.zeros(N_tokens, num_experts)
logits_collapsed[:, 0] = 100.0
loss_collapsed = moe_load_balance_loss(logits_collapsed, num_experts)
print(f"Collapsed routing loss: {loss_collapsed.item():.4f}  (expected > 1.0)")

torch.manual_seed(0)
logits_grad = torch.randn(16, num_experts, requires_grad=True)
loss_grad = moe_load_balance_loss(logits_grad, num_experts)
loss_grad.backward()
print(f"Gradient exists: {logits_grad.grad is not None}")
```

