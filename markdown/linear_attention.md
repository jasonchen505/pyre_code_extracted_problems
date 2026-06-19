# Linear Self-Attention

**中文标题:** 线性自注意力

**难度:** Hard

**函数名:** `linear_attention`

## 英文描述

Implement linear self-attention with kernel feature maps.

Linear attention replaces softmax with a feature map phi, enabling O(S*D^2) complexity instead of O(S^2*D) by reordering the computation.

**Signature:** `linear_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, D_k)
- `K` — key tensor (B, S, D_k)
- `V` — value tensor (B, S, D_v)

**Returns:** attention output (B, S, D_v)

**Constraints:**
- Feature map: `phi(x) = elu(x) + 1`
- Compute `phi(Q) @ (phi(K)^T @ V)` not `softmax(Q @ K^T) @ V`
- Normalize by `phi(Q) @ sum(phi(K))` (add `eps=1e-6` for numerical stability)

## 中文描述

实现基于核特征映射的线性自注意力。

线性注意力用特征映射 phi 替代 softmax，通过重排计算顺序将复杂度从 O(S^2*D) 降至 O(S*D^2)。

**签名:** `linear_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, D_k)
- `K` — 键张量 (B, S, D_k)
- `V` — 值张量 (B, S, D_v)

**返回:** 注意力输出 (B, S, D_v)

**约束:**
- 特征映射：`phi(x) = elu(x) + 1`
- 计算 `phi(Q) @ (phi(K)^T @ V)` 而非 `softmax(Q @ K^T) @ V`
- 通过 `phi(Q) @ sum(phi(K))` 归一化（加 `eps=1e-6` 保证数值稳定）

## 提示

```
`phi(x) = elu(x) + 1`. Let `KV = phi(K).transpose(-2, -1) @ V` and `Z = phi(K).sum(dim=-2, keepdim=True)`. Output = `(phi(Q) @ KV) / (phi(Q) @ Z.transpose(-2, -1) + 1e-6)`.
```

## 参考答案

```python
def linear_attention(Q, K, V):
    Q_prime = torch.where(Q > 0, Q, torch.exp(Q) - 1) + 1
    K_prime = torch.where(K > 0, K, torch.exp(K) - 1) + 1
    KV = torch.bmm(K_prime.transpose(1, 2), V)       # (B, D_k, D_v)
    Z = K_prime.sum(dim=1, keepdim=True)              # (B, 1, D_k)
    num = torch.bmm(Q_prime, KV)                      # (B, S, D_v)
    den = torch.bmm(Q_prime, Z.transpose(1, 2))       # (B, S, 1)
    return num / (den + 1e-6)
```

## 示例代码

```python
Q=torch.randn(1,8,16); K=torch.randn(1,8,16); V=torch.randn(1,8,32)
print('Shape:', linear_attention(Q,K,V).shape)
```

