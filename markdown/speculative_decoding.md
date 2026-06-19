# Speculative Decoding

**中文标题:** 推测解码

**难度:** Hard

**函数名:** `speculative_decode`

## 英文描述

Implement speculative decoding for faster LLM inference.

Speculative decoding uses a fast draft model to propose tokens, then verifies them against the target model, accepting or resampling based on probability ratios.

**Signature:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`

**Parameters:**
- `target_probs` — target model probabilities (K, V)
- `draft_probs` — draft model probabilities (K, V)
- `draft_tokens` — proposed token IDs (K,)

**Returns:** list of accepted token IDs (length 1 to K)

**Constraints:**
- Accept with prob `min(1, p_target/p_draft)`
- On rejection, resample from `max(0, p_target - p_draft)` normalized
- Stop at first rejection (return accepted so far + resampled token)

## 中文描述

实现推测解码以加速 LLM 推理。

推测解码使用快速草稿模型提议 token，然后根据概率比率与目标模型验证，接受或重新采样。

**签名:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`

**参数:**
- `target_probs` — 目标模型概率 (K, V)
- `draft_probs` — 草稿模型概率 (K, V)
- `draft_tokens` — 提议的 token ID (K,)

**返回:** 接受的 token ID 列表（长度 1 到 K）

**约束:**
- 以概率 `min(1, p_target/p_draft)` 接受
- 拒绝时从归一化的 `max(0, p_target - p_draft)` 重新采样
- 在首次拒绝时停止（返回已接受的 + 重采样 token）

## 提示

```
for i in range(K):
  accept_prob = min(1, p_target[i, token] / p_draft[i, token])
  if accepted: append token, continue
  else: resample from max(0, p_target[i] - p_draft[i]) normalized, append, return
```

## 参考答案

```python
def speculative_decode(target_probs, draft_probs, draft_tokens):
    K = len(draft_tokens)
    accepted = []
    for i in range(K):
        t = draft_tokens[i].item()
        ratio = target_probs[i, t] / max(draft_probs[i, t].item(), 1e-10)
        if torch.rand(1).item() < min(1.0, ratio.item()):
            accepted.append(t)
        else:
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            s = adjusted.sum()
            if s > 0:
                adjusted = adjusted / s
            else:
                adjusted = torch.ones_like(adjusted) / adjusted.shape[0]
            accepted.append(torch.multinomial(adjusted, 1).item())
            return accepted
    return accepted
```

## 示例代码

```python
torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
print('Perfect draft:', speculative_decode(probs, probs, tokens))
```

