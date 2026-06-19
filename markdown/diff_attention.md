# Differential Attention

**中文标题:** 差分注意力

**难度:** Hard

**函数名:** `diff_attention`

## 英文描述

Implement Differential Attention from the ICLR 2025 paper "Differential Transformer".

The key idea: split Q and K each into two halves, compute two separate softmax attention maps, then subtract them (scaled by a learnable lambda) to cancel noise and improve focus on relevant context.

**Signature:** `diff_attention(Q, K, V, lambda_val) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, 2*D_h)
- `K` — key tensor (B, S, 2*D_h)
- `V` — value tensor (B, S, D_v)
- `lambda_val` — scalar float or 0-dim tensor controlling noise cancellation

**Returns:** output tensor (B, S, D_v)

**Formula:**
```
out = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V
```

**Constraints:**
- Split Q into Q1, Q2 along last dim; same for K
- Use `torch.softmax(..., dim=-1)` (not F.softmax)
- Scale by 1/√D_h before softmax

## 中文描述

实现 ICLR 2025 论文《Differential Transformer》中的差分注意力机制。

核心思想：将 Q 和 K 各自分成两半，分别计算两个 softmax 注意力图，然后相减（乘以可学习的 lambda）以消除噪声，提升对相关上下文的聚焦能力。

**签名:** `diff_attention(Q, K, V, lambda_val) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, 2*D_h)
- `K` — 键张量 (B, S, 2*D_h)
- `V` — 值张量 (B, S, D_v)
- `lambda_val` — 标量浮点数或 0 维张量，控制噪声消除强度

**返回:** 输出张量 (B, S, D_v)

**公式:**
```
out = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V
```

**约束:**
- 沿最后一维将 Q 拆分为 Q1、Q2；K 同理
- 使用 `torch.softmax(..., dim=-1)`（不能用 F.softmax）
- softmax 前除以 √D_h

## 提示

```
1. `Q1,Q2 = Q[...,:D_h], Q[...,D_h:]`; same for K
2. `scale = D_h**-0.5`
3. `A1 = softmax(Q1@K1.T * scale)`, `A2 = softmax(Q2@K2.T * scale)`
4. `return (A1 - lambda_val*A2) @ V`
```

## 参考答案

```python
def diff_attention(Q, K, V, lambda_val):
    B, S, D2 = Q.shape
    D_h = D2 // 2
    Q1, Q2 = Q[..., :D_h], Q[..., D_h:]
    K1, K2 = K[..., :D_h], K[..., D_h:]
    scale = D_h ** -0.5
    A1 = torch.softmax(Q1 @ K1.transpose(-2, -1) * scale, dim=-1)
    A2 = torch.softmax(Q2 @ K2.transpose(-2, -1) * scale, dim=-1)
    return (A1 - lambda_val * A2) @ V
```

## 示例代码

```python
torch.manual_seed(0)
B, S, D2, D_v = 2, 4, 8, 6
Q = torch.randn(B, S, D2)
K = torch.randn(B, S, D2)
V = torch.randn(B, S, D_v)

D_h = D2 // 2
scale = D_h ** -0.5
standard = torch.softmax(Q[..., :D_h] @ K[..., :D_h].transpose(-2, -1) * scale, dim=-1) @ V
diff_zero = diff_attention(Q, K, V, lambda_val=0.0)
print("lambda=0 matches standard attention:", torch.allclose(diff_zero, standard, atol=1e-6))

Q_same = torch.cat([Q[..., :D_h], Q[..., :D_h]], dim=-1)
K_same = torch.cat([K[..., :D_h], K[..., :D_h]], dim=-1)
diff_one = diff_attention(Q_same, K_same, V, lambda_val=1.0)
print("lambda=1 with identical halves gives zero:", torch.allclose(diff_one, torch.zeros_like(diff_one), atol=1e-6))
print("Output shape:", diff_zero.shape)  # (2, 4, 6)
```

