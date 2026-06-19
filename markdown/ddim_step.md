# DDIM Sampling Step

**中文标题:** DDIM 采样步骤

**难度:** Medium

**函数名:** `ddim_step`

## 英文描述

Implement one step of DDIM (Denoising Diffusion Implicit Models) deterministic sampling.

Given the current noisy sample and the model's noise prediction, compute the previous (less noisy) sample without any stochastic noise injection.

**Signature:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`

**Parameters:**
- `x_t` — current noisy sample at timestep t, shape (B, ...)
- `noise_pred` — model's predicted noise ε_θ(x_t, t), same shape as x_t
- `alpha_bar_t` — scalar, cumulative noise schedule ᾱ_t at current step
- `alpha_bar_prev` — scalar, cumulative noise schedule ᾱ_{t-1} at previous step

**Returns:** x_{t-1} with same shape as x_t

**Formula:**
1. Predict x0: `x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`
2. Direction toward x_t: `noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`
3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`

## 中文描述

实现 DDIM（去噪扩散隐式模型）的单步确定性采样。

给定当前含噪样本和模型的噪声预测，在不注入随机噪声的情况下计算上一步（噪声更少的）样本。

**签名:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`

**参数:**
- `x_t` — 时间步 t 的含噪样本，形状 (B, ...)
- `noise_pred` — 模型预测的噪声 ε_θ(x_t, t)，形状与 x_t 相同
- `alpha_bar_t` — 标量，当前步的累积噪声调度 ᾱ_t
- `alpha_bar_prev` — 标量，上一步的累积噪声调度 ᾱ_{t-1}

**返回:** x_{t-1}，形状与 x_t 相同

**公式:**
1. 预测 x0：`x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`
2. 指向 x_t 的方向：`noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`
3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`

## 提示

```
1. x0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
2. noise_dir = √(1-ᾱ_{t-1})·ε
3. x_{t-1} = √ᾱ_{t-1}·x0_pred + noise_dir
   Use ** 0.5 for square roots.
```

## 参考答案

```python
def ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev):
    # Predict x0
    x0_pred = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
    # Direction toward x_t
    noise_direction = (1 - alpha_bar_prev) ** 0.5 * noise_pred
    # Previous sample
    x_prev = (alpha_bar_prev ** 0.5) * x0_pred + noise_direction
    return x_prev
```

## 示例代码

```python
torch.manual_seed(42)

T = 5
alpha_bars = torch.linspace(0.99, 0.01, T + 1)  # index 0..T

x_clean = torch.tensor([1.0, -1.0, 0.5])  # target signal
noise   = torch.randn_like(x_clean)

ab_T = alpha_bars[T]
x_t  = ab_T ** 0.5 * x_clean + (1 - ab_T) ** 0.5 * noise

print(f"{'Step':>4}  {'alpha_bar_t':>12}  {'x_t (first elem)':>18}")
print("-" * 42)
for step in range(T, 0, -1):
    ab_t    = alpha_bars[step]
    ab_prev = alpha_bars[step - 1]
    noise_pred = (x_t - ab_t ** 0.5 * x_clean) / (1 - ab_t) ** 0.5
    x_t = ddim_step(x_t, noise_pred, ab_t, ab_prev)
    print(f"{step:>4}  {ab_prev.item():>12.4f}  {x_t[0].item():>18.4f}")

print(f"
Final x vs clean: {x_t.tolist()}  vs  {x_clean.tolist()}")
```

