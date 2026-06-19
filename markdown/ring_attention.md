# Ring Attention

**中文标题:** 环形注意力

**难度:** Hard

**函数名:** `ring_attention`

## 英文描述

Implement ring attention (single-device simulation).

Ring attention enables sequence parallelism by partitioning Q/K/V across devices in a ring. Each device holds a Q chunk and rotates K/V chunks around the ring, accumulating attention outputs using online softmax.

**Signature:** `ring_attention(Q, K, V, num_devices) -> Tensor`

**Parameters:**
- `Q, K, V` — tensors of shape `(B, S, D)`, S divisible by num_devices
- `num_devices` — number of virtual ring participants

**Returns:** output tensor `(B, S, D)`, numerically equivalent to standard attention

**Algorithm:**
1. Split Q, K, V into `num_devices` chunks along S
2. For each Q chunk (device i), iterate over all K/V chunks in ring order
3. Accumulate using online softmax (track running max and sum)
4. Reassemble output chunks

## 中文描述

实现环形注意力（单机模拟）。

环形注意力通过将 Q/K/V 分布在环形拓扑的设备上实现序列并行。每个设备持有一个 Q 分块，K/V 分块在环上轮转，用 online softmax 累加注意力输出。

**签名:** `ring_attention(Q, K, V, num_devices) -> Tensor`

**参数:**
- `Q, K, V` — 形状 `(B, S, D)` 的张量，S 可被 num_devices 整除
- `num_devices` — 虚拟环参与者数量

**返回:** 输出张量 `(B, S, D)`，数值上等价于标准注意力

**算法:**
1. 沿 S 维将 Q, K, V 分成 `num_devices` 个分块
2. 对每个 Q 分块（设备 i），遍历所有 K/V 分块（环形顺序）
3. 用 online softmax 累加（跟踪运行最大值和归一化因子）
4. 重新拼接输出分块

## 提示

```
Split Q/K/V into num_devices chunks. For each Q chunk i, loop over all K/V chunks (ring order).
Online softmax per step:
  m_new = max(m, scores.max)
  l_new = l·exp(m-m_new) + Σexp(scores-m_new)
  o = (o·l·exp(m-m_new) + exp(scores-m_new)@V) / l_new
```

## 参考答案

```python
def ring_attention(Q, K, V, num_devices):
    B, S, D = Q.shape
    chunk = S // num_devices
    scale = D ** -0.5

    Q_chunks = Q.split(chunk, dim=1)   # list of (B, chunk, D)
    K_chunks = K.split(chunk, dim=1)
    V_chunks = V.split(chunk, dim=1)

    outputs = []
    for i, Qi in enumerate(Q_chunks):
        # Online softmax accumulators
        m = torch.full((B, chunk, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
        l = torch.zeros(B, chunk, 1, device=Q.device, dtype=Q.dtype)
        o = torch.zeros(B, chunk, D, device=Q.device, dtype=Q.dtype)

        for j in range(num_devices):
            Kj = K_chunks[(i + j) % num_devices]
            Vj = V_chunks[(i + j) % num_devices]
            scores = (Qi @ Kj.transpose(-2, -1)) * scale  # (B, chunk, chunk)
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            exp_scores = torch.exp(scores - m_new)
            l_new = l * torch.exp(m - m_new) + exp_scores.sum(dim=-1, keepdim=True)
            o = o * (l * torch.exp(m - m_new)) / l_new + (exp_scores @ Vj) / l_new
            m, l = m_new, l_new

        outputs.append(o)

    return torch.cat(outputs, dim=1)
```

## 示例代码

```python
torch.manual_seed(42)
B, S, D = 2, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)

scale = D ** -0.5
scores_ref = (Q @ K.transpose(-2, -1)) * scale
ref_out = torch.softmax(scores_ref, dim=-1) @ V

for num_devices in [2, 4]:
    ring_out = ring_attention(Q, K, V, num_devices=num_devices)
    max_diff = (ring_out - ref_out).abs().max().item()
    match = torch.allclose(ring_out, ref_out, atol=1e-5)
    print(f'num_devices={num_devices}  shape={tuple(ring_out.shape)}  max_diff={max_diff:.2e}  match={match}')
```

