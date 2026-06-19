# Adaptive LayerNorm Zero (adaLN-Zero)

**中文标题:** 自适应层归一化 Zero

**难度:** Medium

**函数名:** `adaln_zero`

## 英文描述

Implement the adaLN-Zero conditioning mechanism from DiT (Diffusion Transformer, ICCV 2023).

A linear projection of the conditioning embedding (timestep + class label) regresses scale (γ), shift (β), and gate (α) parameters. These modulate the layer-normed input. The gate is zero-initialized so the block starts as an identity function.

**Signature:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`

**Parameters:**
- `x` — input tokens (B, N, D)
- `cond` — conditioning embedding (B, D)
- `W_ada` — linear weight (D, 6*D), regresses [γ1, β1, α1, γ2, β2, α2]
- `b_ada` — bias (6*D,), zero-initialized in practice

**Returns:** modulated tensor (B, N, D) using the first set of params (γ1, β1, α1)

**Steps:**
1. `params = cond @ W_ada + b_ada` — shape (B, 6*D)
2. Split into 6 chunks of size D: γ1, β1, α1, γ2, β2, α2
3. LayerNorm x manually (no learnable affine): subtract mean, divide by std
4. Modulate: `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`
5. Return out

## 中文描述

实现 DiT（扩散变换器，ICCV 2023）中的 adaLN-Zero 条件机制。

通过对条件嵌入（时间步 + 类别标签）进行线性投影，回归出缩放（γ）、偏移（β）和门控（α）参数，用于调制经过层归一化的输入。门控参数零初始化，使得模块初始时等价于恒等映射。

**签名:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`

**参数:**
- `x` — 输入 token (B, N, D)
- `cond` — 条件嵌入 (B, D)
- `W_ada` — 线性权重 (D, 6*D)，回归 [γ1, β1, α1, γ2, β2, α2]
- `b_ada` — 偏置 (6*D,)，实践中零初始化

**返回:** 使用第一组参数 (γ1, β1, α1) 调制后的张量 (B, N, D)

**步骤:**
1. `params = cond @ W_ada + b_ada` — 形状 (B, 6*D)
2. 沿最后一维切分为 6 块，每块大小 D：γ1, β1, α1, γ2, β2, α2
3. 手动计算 LayerNorm（无可学习仿射参数）：减均值，除以标准差
4. 调制：`out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`
5. 返回 out

## 提示

```
1. `params = cond @ W_ada + b_ada`  shape `(B, 6*D)`
2. `γ1,β1,α1,... = params.chunk(6, dim=-1)`
3. Manual LayerNorm: `mean/var` over `dim=-1`, `unbiased=False`, `eps=1e-6`
4. `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`
```

## 参考答案

```python
def adaln_zero(x, cond, W_ada, b_ada):
    B, N, D = x.shape
    params = cond @ W_ada + b_ada          # (B, 6*D)
    chunks = params.chunk(6, dim=-1)       # 6 x (B, D)
    gamma1, beta1, alpha1 = chunks[0], chunks[1], chunks[2]
    # LayerNorm manually
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / (var + 1e-6).sqrt()
    # Modulate: unsqueeze for broadcast over N tokens
    out = alpha1.unsqueeze(1) * (gamma1.unsqueeze(1) * x_norm + beta1.unsqueeze(1))
    return out
```

## 示例代码

```python
torch.manual_seed(0)

B, N, D = 4, 16, 32
C = 16  # conditioning dim

x    = torch.randn(B, N, D)
cond = torch.randn(B, C)

W_ada = torch.zeros(C, 6 * D)
b_ada = torch.zeros(6 * D)
out_zero = adaln_zero(x, cond, W_ada, b_ada)
print(f"Zero W_ada, zero b_ada => max abs output: {out_zero.abs().max().item():.6f}  (expected 0.0)")

W_ada_rand = torch.randn(C, 6 * D) * 0.1
b_ada_rand = torch.randn(6 * D) * 0.1
out_rand = adaln_zero(x, cond, W_ada_rand, b_ada_rand)
print(f"Random W_ada          => output shape: {out_rand.shape}, mean abs: {out_rand.abs().mean().item():.4f}")
```

