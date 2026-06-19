# Top-k / Top-p Sampling

**中文标题:** Top-k / Top-p 采样

**难度:** Medium

**函数名:** `sample_top_k_top_p`

## 英文描述

Implement top-k / top-p (nucleus) sampling for language model decoding.

These sampling strategies filter the vocabulary to high-probability tokens before sampling, balancing diversity and quality in text generation.

**Signature:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`

**Parameters:**
- `logits` — raw logits over vocabulary (V,)
- `top_k` — keep only top-k tokens (0 = disabled)
- `top_p` — keep tokens with cumulative prob <= p (1.0 = disabled)
- `temperature` — temperature scaling

**Returns:** sampled token index (int)

**Constraints:**
- Apply temperature first, then top-k, then top-p
- `top_k=1` must always return argmax

## 中文描述

实现语言模型解码的 top-k / top-p（核）采样。

这些采样策略在采样前将词表过滤为高概率 token，在文本生成中平衡多样性和质量。

**签名:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`

**参数:**
- `logits` — 词表上的原始 logits (V,)
- `top_k` — 仅保留 top-k 个 token（0 = 禁用）
- `top_p` — 保留累积概率 <= p 的 token（1.0 = 禁用）
- `temperature` — 温度缩放

**返回:** 采样的 token 索引（整数）

**约束:**
- 先应用温度，再 top-k，再 top-p
- `top_k=1` 必须始终返回 argmax

## 提示

```
1. logits /= temperature
2. top-k: set logits below k-th largest to -inf
3. top-p: sort desc → cumsum of softmax probs → mask where cumsum > p → set to -inf
4. sample from softmax(logits)
```

## 参考答案

```python
def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
```

## 示例代码

```python
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))
```

