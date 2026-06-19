# Flash Attention (Tiled)

**中文标题:** Flash Attention（分块）

**难度:** Hard

**函数名:** `flash_attention`

## 英文描述

Implement tiled (flash) attention with online softmax.

Flash attention processes Q/K/V in blocks to reduce memory usage, using the online softmax trick to maintain numerical correctness across tiles.

**Signature:** `flash_attention(Q, K, V, block_size=32) -> Tensor`

**Parameters:**
- `Q`, `K`, `V` — input tensors (B, S, D)
- `block_size` — tile size for blocking

**Returns:** attention output (B, S, D), identical to standard attention

**Constraints:**
- Must match standard softmax attention numerically
- Handle non-aligned sequence lengths (S not divisible by block_size)
- Result must be invariant to block_size choice

## 中文描述

实现分块（Flash）注意力与在线 softmax。

Flash 注意力将 Q/K/V 分块处理以减少内存使用，利用在线 softmax 技巧在分块间保持数值正确性。

**签名:** `flash_attention(Q, K, V, block_size=32) -> Tensor`

**参数:**
- `Q`, `K`, `V` — 输入张量 (B, S, D)
- `block_size` — 分块大小

**返回:** 注意力输出 (B, S, D)，与标准注意力数值一致

**约束:**
- 必须与标准 softmax 注意力数值匹配
- 处理非对齐序列长度（S 不能被 block_size 整除）
- 结果不受 block_size 选择影响

## 提示

```
Process Q in tiles. For each Q-block, iterate over K/V blocks:
1. `block_max = scores.max(dim=-1)` → `new_max = max(row_max, block_max)`
2. `correction = exp(row_max - new_max)` → rescale `acc` and `row_sum`
3. `acc += exp(scores - new_max) @ V_block`
4. `output = acc / row_sum`
```

## 参考答案

```python
def flash_attention(Q, K, V, block_size=32):
    B, S, D = Q.shape
    output = torch.zeros_like(Q)
    for i in range(0, S, block_size):
        qi = Q[:, i:i+block_size]
        bs_q = qi.shape[1]
        row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
        row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
        acc = torch.zeros(B, bs_q, D, device=Q.device)
        for j in range(0, S, block_size):
            kj = K[:, j:j+block_size]
            vj = V[:, j:j+block_size]
            scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            acc = acc * correction + torch.bmm(exp_scores, vj)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max
        output[:, i:i+block_size] = acc / row_sum
    return output
```

## 示例代码

```python
Q, K, V = torch.randn(1, 16, 8), torch.randn(1, 16, 8), torch.randn(1, 16, 8)
out = flash_attention(Q, K, V, block_size=4)
scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(8)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
print('Shape:', out.shape)
print('Max diff:', (out - ref).abs().max().item())
```

