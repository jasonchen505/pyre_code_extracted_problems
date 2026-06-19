# Multi-Token Prediction Loss

**中文标题:** 多 Token 预测

**难度:** Medium

**函数名:** `multi_token_prediction_loss`

## 英文描述

Implement the Multi-Token Prediction (MTP) training loss from Meta's 2024 paper.

Instead of predicting only the next token, train N independent prediction heads to simultaneously predict the next N tokens from the same hidden states. Average the cross-entropy loss across all heads.

**Signature:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`

**Parameters:**
- `hidden_states` — transformer hidden states (B, S, D)
- `heads` — list of N weight matrices, each (D, vocab_size)
- `targets` — target token IDs (B, S, N), where `targets[:, :, i]` are the labels for head i

**Returns:** scalar mean loss (average CE across all N heads)

**Constraints:**
- Implement cross-entropy manually: log-softmax then gather
- Do NOT use F.cross_entropy or F.log_softmax
- Use numerically stable log-softmax: subtract max before exp

## 中文描述

实现 Meta 2024 年论文中的多 Token 预测（MTP）训练损失。

不只预测下一个 token，而是训练 N 个独立的预测头，从相同的隐藏状态同时预测接下来的 N 个 token。对所有头的交叉熵损失取平均。

**签名:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`

**参数:**
- `hidden_states` — Transformer 隐藏状态 (B, S, D)
- `heads` — N 个权重矩阵的列表，每个形状为 (D, vocab_size)
- `targets` — 目标 token ID (B, S, N)，其中 `targets[:, :, i]` 是第 i 个头的标签

**返回:** 标量均值损失（所有 N 个头的 CE 均值）

**约束:**
- 手动实现交叉熵：log-softmax 后 gather
- 不能使用 F.cross_entropy 或 F.log_softmax
- 使用数值稳定的 log-softmax：exp 前先减去最大值

## 提示

```
For each head `i`:
1. `logits = hidden_states @ heads[i]`
2. Stable log-softmax: `shifted = logits - logits.max(-1,keepdim=True).values`
   `log_probs = shifted - log(exp(shifted).sum(-1,keepdim=True))`
3. `log_p = log_probs.gather(-1, targets[:,:,i].unsqueeze(-1)).squeeze(-1)`
4. `loss += -log_p.mean()`
Return `total_loss / N`
```

## 参考答案

```python
def multi_token_prediction_loss(hidden_states, heads, targets):
    B, S, D = hidden_states.shape
    N = len(heads)
    total_loss = 0.0
    for i, head in enumerate(heads):
        logits = hidden_states @ head          # (B, S, vocab_size)
        # Numerically stable log-softmax
        logits_max = logits.max(dim=-1, keepdim=True).values
        shifted = logits - logits_max
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
        # Gather log-prob of target tokens
        tgt = targets[:, :, i]                 # (B, S)
        log_p = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total_loss = total_loss + (-log_p.mean())
    return total_loss / N
```

## 示例代码

```python
torch.manual_seed(0)
B, S, D, V = 2, 5, 16, 10

hidden = torch.randn(B, S, D)
head_single = torch.randn(D, V)
targets_single = torch.randint(0, V, (B, S, 1))

mtp_loss = multi_token_prediction_loss(hidden, [head_single], targets_single)

logits = hidden @ head_single
ce_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets_single[:, :, 0].reshape(-1))

print(f"MTP loss (N=1):  {mtp_loss.item():.6f}")
print(f"Standard CE:     {ce_loss.item():.6f}")
print(f"N=1 matches CE:  {torch.allclose(mtp_loss, ce_loss, atol=1e-5)}")

heads_3 = [torch.randn(D, V) for _ in range(3)]
targets_3 = torch.randint(0, V, (B, S, 3))
loss_3 = multi_token_prediction_loss(hidden, heads_3, targets_3)
print(f"MTP loss (N=3):  {loss_3.item():.6f}")
```

