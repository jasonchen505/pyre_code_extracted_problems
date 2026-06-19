# Sliding Window Attention

**中文标题:** 滑动窗口注意力

**难度:** Medium

**函数名:** `sliding_window_attention`

## 英文描述

Implement sliding window attention.

Sliding window attention restricts each position to attend only within a fixed window, reducing complexity for long sequences while maintaining local context.

**Signature:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`

**Parameters:**
- `Q`, `K`, `V` — input tensors (B, S, D)
- `window_size` — each position attends to positions within |i-j| <= window_size

**Returns:** attention output (B, S, D)

**Constraints:**
- Mask positions where `|i - j| > window_size` with `-inf`
- `window_size=0` means each position only sees itself
- Large window equals full attention

## 中文描述

实现滑动窗口注意力。

滑动窗口注意力限制每个位置只关注固定窗口内的位置，在保持局部上下文的同时降低长序列的复杂度。

**签名:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`

**参数:**
- `Q`, `K`, `V` — 输入张量 (B, S, D)
- `window_size` — 每个位置关注 |i-j| <= window_size 范围内的位置

**返回:** 注意力输出 (B, S, D)

**约束:**
- 用 `-inf` 掩盖 `|i - j| > window_size` 的位置
- `window_size=0` 表示每个位置只能看到自身
- 大窗口等同于全注意力

## 提示

```
Standard attention + window mask. Build `|i-j|` matrix → mask positions where `|i-j| > window_size` with `-inf` → softmax → `@ V`.
```

## 参考答案

```python
def sliding_window_attention(Q, K, V, window_size):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

## 示例代码

```python
Q=torch.randn(1,6,8); K=torch.randn(1,6,8); V=torch.randn(1,6,8)
print('window=0==V?', torch.allclose(sliding_window_attention(Q,K,V,0), V, atol=1e-5))
```

