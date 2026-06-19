# Contrastive Loss (InfoNCE)

**中文标题:** 对比损失（InfoNCE）

**难度:** Medium

**函数名:** `contrastive_loss`

## 英文描述

Implement InfoNCE / NT-Xent contrastive loss, the objective used in CLIP and SimCLR.

Given a batch of query-key pairs, each query `q[i]` should be pulled toward its matching key `k[i]` and pushed away from all other keys.

**Signature:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`

**Parameters:**
- `q` — query embeddings, shape (N, D), assumed L2-normalized
- `k` — key embeddings, shape (N, D), assumed L2-normalized; `k[i]` is the positive for `q[i]`
- `temperature` — softmax temperature τ

**Returns:** scalar mean loss

**Formula:**
```
L = mean over i of: -log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```
This is equivalent to cross-entropy with targets `[0, 1, 2, ..., N-1]`.

**Constraints:**
- Do not use `F.*` or `nn.*` for cross-entropy or softmax
- Implement the log-sum-exp manually

## 中文描述

实现 InfoNCE / NT-Xent 对比损失，即 CLIP 和 SimCLR 中使用的训练目标。

给定一批查询-键对，每个查询 `q[i]` 应被拉近其匹配键 `k[i]`，并远离所有其他键。

**签名:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`

**参数:**
- `q` — 查询嵌入，形状 (N, D)，假设已 L2 归一化
- `k` — 键嵌入，形状 (N, D)，假设已 L2 归一化；`k[i]` 是 `q[i]` 的正样本
- `temperature` — softmax 温度 τ

**返回:** 标量均值损失

**公式:**
```
L = 对 i 求均值：-log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```
等价于目标为 `[0, 1, 2, ..., N-1]` 的交叉熵。

**约束:**
- 不得使用 `F.*` 或 `nn.*` 计算交叉熵或 softmax
- 手动实现 log-sum-exp

## 提示

```
1. `logits = (q @ k.T) / temperature`  shape `(N, N)`
2. `log_sum_exp = logits.logsumexp(dim=-1)`
3. `pos = logits[arange(N), arange(N)]`
4. `return -(pos - log_sum_exp).mean()`
```

## 参考答案

```python
def contrastive_loss(q, k, temperature=0.07):
    # q, k: (N, D), assumed L2-normalized
    logits = (q @ k.T) / temperature  # (N, N)
    N = q.shape[0]
    # manual cross-entropy: log_p = logit_ii - log(sum_j exp(logit_ij))
    log_sum_exp = logits.logsumexp(dim=-1)  # (N,)
    positive_logits = logits[torch.arange(N, device=q.device), torch.arange(N, device=q.device)]
    log_p = positive_logits - log_sum_exp
    return -log_p.mean()
```

## 示例代码

```python
torch.manual_seed(0)
N, D = 8, 16

v = torch.randn(N, D)
q = v / v.norm(dim=-1, keepdim=True)
k = q.clone()
loss_perfect = contrastive_loss(q, k)
print(f"Perfect alignment loss: {loss_perfect:.4f}  (should be near log(1/N) = {-torch.log(torch.tensor(N, dtype=torch.float)):.4f})")

q_rand = torch.randn(N, D)
q_rand = q_rand / q_rand.norm(dim=-1, keepdim=True)
k_rand = torch.randn(N, D)
k_rand = k_rand / k_rand.norm(dim=-1, keepdim=True)
loss_rand = contrastive_loss(q_rand, k_rand)
print(f"Random embeddings loss:  {loss_rand:.4f}  (should be higher)")
```

