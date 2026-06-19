# Mamba SSM Step

**中文标题:** Mamba SSM 步骤

**难度:** Hard

**函数名:** `mamba_ssm_step`

## 英文描述

Implement a single recurrent step of the Mamba selective state space model (SSM).

Mamba's key innovation is that B, C, and delta are **input-dependent** (selective), unlike classical SSMs where they are fixed. This allows the model to selectively remember or forget information.

**Signature:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`

**Parameters:**
- `x` — input at current step, shape (B, D)
- `h` — hidden state, shape (B, D, N)
- `A` — state transition, shape (D, N) — used as `A = -exp(A_log)` upstream
- `B` — input projection (selective), shape (B, D, N)
- `C` — output projection (selective), shape (B, D, N)
- `delta` — discretization step (selective, after softplus), shape (B, D)

**Discretization (Zero-Order Hold):**
- `dA = exp(delta.unsqueeze(-1) * A)` — shape (B, D, N)
- `dB = delta.unsqueeze(-1) * B` — shape (B, D, N)

**Recurrence:**
- `h_new = dA * h + dB * x.unsqueeze(-1)`
- `y = (h_new * C).sum(dim=-1)` — shape (B, D)

**Returns:** tuple `(y, h_new)`

## 中文描述

实现 Mamba 选择性状态空间模型（SSM）的单步递推。

Mamba 的核心创新在于 B、C 和 delta 是**输入相关的**（选择性的），不同于经典 SSM 中的固定参数。这使模型能够选择性地记忆或遗忘信息。

**签名:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`

**参数:**
- `x` — 当前步输入，形状 (B, D)
- `h` — 隐藏状态，形状 (B, D, N)
- `A` — 状态转移矩阵，形状 (D, N)（上游使用 `A = -exp(A_log)`）
- `B` — 输入投影（选择性），形状 (B, D, N)
- `C` — 输出投影（选择性），形状 (B, D, N)
- `delta` — 离散化步长（选择性，经过 softplus），形状 (B, D)

**离散化（零阶保持）:**
- `dA = exp(delta.unsqueeze(-1) * A)` — 形状 (B, D, N)
- `dB = delta.unsqueeze(-1) * B` — 形状 (B, D, N)

**递推:**
- `h_new = dA * h + dB * x.unsqueeze(-1)`
- `y = (h_new * C).sum(dim=-1)` — 形状 (B, D)

**返回:** 元组 `(y, h_new)`

## 提示

```
1. `dA = exp(delta.unsqueeze(-1) * A)`  shape `(B,D,N)`
2. `dB = delta.unsqueeze(-1) * B`
3. `h_new = dA * h + dB * x.unsqueeze(-1)`
4. `y = (h_new * C).sum(dim=-1)`  shape `(B,D)`
```

## 参考答案

```python
def mamba_ssm_step(x, h, A, B, C, delta):
    # delta: (B, D), A: (D, N), B: (B, D, N), C: (B, D, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))   # (B, D, N)
    dB = delta.unsqueeze(-1) * B                            # (B, D, N)
    h_new = dA * h + dB * x.unsqueeze(-1)                  # (B, D, N)
    y = (h_new * C).sum(dim=-1)                             # (B, D)
    return y, h_new
```

## 示例代码

```python
torch.manual_seed(0)
B_size, D, N = 2, 8, 4  # batch, channels, state_dim

x     = torch.randn(B_size, D)
h     = torch.zeros(B_size, D, N)
A     = -torch.rand(D, N)          # negative for stability
B_mat = torch.randn(D, N)
C     = torch.randn(B_size, D, N)
delta = torch.rand(B_size, D).add(0.1)  # positive step sizes

y, h_new = mamba_ssm_step(x, h, A, B_mat, C, delta)

print("y shape:    ", y.shape)      # (2, 8)
print("h_new shape:", h_new.shape)  # (2, 8, 4)

dA_manual = torch.exp(delta[0, 0] * A[0])
dA_check  = torch.exp(delta[0:1, 0:1] * A[0:1])[0, 0]
print("dA formula check (should be ~0):", (dA_manual - dA_check).abs().max().item())
```

