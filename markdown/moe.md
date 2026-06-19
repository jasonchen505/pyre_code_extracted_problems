# Mixture of Experts (MoE)

**中文标题:** 混合专家（MoE）

**难度:** Hard

**函数名:** `MixtureOfExperts`

## 英文描述

Implement a Mixture of Experts (MoE) layer as an nn.Module.

MoE routes each token to the top-k experts via a learned router, combining their outputs with softmax-normalized weights for conditional computation.

**Signature:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, S, d_model)

**Returns:** output tensor (B, S, d_model)

**Constraints:**
- Router: `nn.Linear(d_model, num_experts)` -> topk -> softmax
- Each expert: Linear -> ReLU -> Linear
- Store experts in `self.experts` (nn.ModuleList)

## 中文描述

实现混合专家（MoE）层（nn.Module）。

MoE 通过学习的路由器将每个 token 路由到 top-k 个专家，用 softmax 归一化的权重组合它们的输出，实现条件计算。

**签名:** `MixtureOfExperts(d_model, d_ff, num_experts, top_k=2)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, S, d_model)

**返回:** 输出张量 (B, S, d_model)

**约束:**
- 路由器：`nn.Linear(d_model, num_experts)` -> topk -> softmax
- 每个专家：Linear -> ReLU -> Linear
- 专家存储在 `self.experts`（nn.ModuleList）中

## 提示

```
`router`: `Linear(d, num_experts)` → `topk` → `softmax` weights. Each expert: `Linear → ReLU → Linear`. Weighted sum of top-k expert outputs per token.
```

## 参考答案

```python
class _ManualReLU(nn.Module):
    def forward(self, x):
        return x.clamp(min=0)

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), _ManualReLU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(-1, D)
        else:
            x_flat = x
        logits = self.router(x_flat)
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    output[mask] += weights[mask, k:k+1] * self.experts[e](x_flat[mask])
        return output.reshape(orig_shape)
```

## 示例代码

```python
moe = MixtureOfExperts(32, 64, num_experts=4, top_k=2)
x = torch.randn(2, 8, 32)
print('Output:', moe(x).shape)
print('Params:', sum(p.numel() for p in moe.parameters()))
```

