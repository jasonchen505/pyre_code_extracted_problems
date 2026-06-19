# Flow Matching Loss

**中文标题:** 流匹配损失

**难度:** Easy

**函数名:** `flow_matching_loss`

## 英文描述

Implement the flow matching training loss, the generative modeling paradigm behind Stable Diffusion 3, Flux, and Sora.

Flow matching trains a neural network to predict the velocity field that transports noise to data along straight paths. The target velocity at any interpolated point is simply the direction from noise to data.

**Signature:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`

**Parameters:**
- `model_output` — predicted velocity v_θ(x_t, t), shape (B, D)
- `x0` — noise samples, shape (B, D)
- `x1` — data samples, shape (B, D)
- `t` — timesteps in [0, 1], shape (B,)

**Returns:** scalar MSE loss

**Formula:**
- Target velocity: `u_t = x1 - x0` (straight-line direction from noise to data)
- Loss: `mean(||model_output - u_t||²)`

Note: the interpolated point `x_t = t * x1 + (1 - t) * x0` is computed externally before calling the model; `t` is passed here only for interface completeness.

## 中文描述

实现流匹配训练损失——Stable Diffusion 3、Flux 和 Sora 背后的生成建模范式。

流匹配训练神经网络预测速度场，将噪声沿直线路径传输到数据。在任意插值点处，目标速度就是从噪声指向数据的方向。

**签名:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`

**参数:**
- `model_output` — 预测速度 v_θ(x_t, t)，形状 (B, D)
- `x0` — 噪声样本，形状 (B, D)
- `x1` — 数据样本，形状 (B, D)
- `t` — [0, 1] 内的时间步，形状 (B,)

**返回:** 标量 MSE 损失

**公式:**
- 目标速度：`u_t = x1 - x0`（从噪声到数据的直线方向）
- 损失：`mean(||model_output - u_t||²)`

注意：插值点 `x_t = t * x1 + (1 - t) * x0` 在调用模型前已在外部计算；`t` 在此仅为接口完整性而传入。

## 提示

```
target = x1 - x0  (straight-line velocity from noise to data)
loss = ((model_output - target) ** 2).mean()
`t` is not used in the loss itself.
```

## 参考答案

```python
def flow_matching_loss(model_output, x0, x1, t):
    target_velocity = x1 - x0
    diff = model_output - target_velocity
    return (diff * diff).mean()
```

## 示例代码

```python
torch.manual_seed(0)

B, D = 16, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t  = torch.rand(B)

perfect_output = x1 - x0
loss_perfect = flow_matching_loss(perfect_output, x0, x1, t)
print(f"Perfect prediction => loss = {loss_perfect.item():.6f}  (expected 0.0)")

random_output = torch.randn(B, D)
loss_random = flow_matching_loss(random_output, x0, x1, t)
print(f"Random prediction  => loss = {loss_random.item():.4f}   (expected > 0)")

noisy_output = perfect_output + 0.1 * torch.randn(B, D)
loss_noisy = flow_matching_loss(noisy_output, x0, x1, t)
print(f"Noisy prediction   => loss = {loss_noisy.item():.4f}   (expected small but > 0)")
```

