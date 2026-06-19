# Mixed Precision Training Step

**中文标题:** 混合精度训练步骤

**难度:** Medium

**函数名:** `mixed_precision_step`

## 英文描述

Implement one mixed precision training step.

Mixed precision training uses fp16 for the forward/backward pass (faster, less memory) while keeping fp32 master weights for numerical stability. Loss scaling prevents fp16 gradient underflow.

**Signature:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`

**Parameters:**
- `model` — nn.Module with fp32 parameters
- `optimizer` — optimizer holding fp32 params
- `loss_fn` — callable `(output, target) -> scalar loss`
- `x, y` — input and target tensors
- `loss_scale` — scalar to multiply loss before backward

**Returns:** unscaled loss value as float

**Steps:**
1. Cast model to fp16, run forward pass
2. Compute loss, scale it by `loss_scale`
3. Run backward on scaled loss
4. Unscale gradients (divide by `loss_scale`)
5. Update optimizer (fp32 master weights)
6. Restore model to fp32
7. Return original (unscaled) loss

## 中文描述

实现一步混合精度训练。

混合精度训练用 fp16 做前向/反向传播（更快、更省内存），同时保留 fp32 主权重以保证数值稳定性。Loss scaling 防止 fp16 梯度下溢。

**签名:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`

**参数:**
- `model` — 持有 fp32 参数的 nn.Module
- `optimizer` — 持有 fp32 参数的优化器
- `loss_fn` — 可调用对象 `(output, target) -> 标量损失`
- `x, y` — 输入和目标张量
- `loss_scale` — 反向传播前乘以 loss 的缩放系数

**返回:** 未缩放的 loss 值（float）

**步骤:**
1. 将模型转为 fp16，运行前向传播
2. 计算 loss，乘以 `loss_scale`
3. 对缩放后的 loss 做反向传播
4. 反缩放梯度（除以 `loss_scale`）
5. 更新优化器（fp32 主权重）
6. 将模型恢复为 fp32
7. 返回原始（未缩放）loss

## 提示

```
1. model.half() → forward with fp16 input
2. compute loss → scale by loss_scale → backward
3. divide all param.grad by loss_scale (unscale)
4. model.float() → optimizer.step()
5. return unscaled loss as float
```

## 参考答案

```python
def mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0):
    # 1. Cast to fp16 for forward pass
    model.half()
    x_fp16 = x.half()
    with torch.no_grad():
        pass  # weights already fp16
    output = model(x_fp16)
    loss = loss_fn(output.float(), y)
    loss_val = loss.item()

    # 2. Scale loss and backward
    optimizer.zero_grad()
    (loss * loss_scale).backward()

    # 3. Unscale gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data.float() / loss_scale

    # 4. Update (fp32 master weights via optimizer)
    model.float()
    optimizer.step()

    return loss_val
```

## 示例代码

```python
torch.manual_seed(42)
model = nn.Linear(8, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(4, 8)
y = torch.randn(4, 4)

weights_before = model.weight.data.clone()

loss_val = mixed_precision_step(model, optimizer, loss_fn, x, y)

print("Loss:", loss_val)
print("Weights updated:", not torch.allclose(model.weight.data, weights_before))
print("Model dtype after step:", model.weight.dtype)
```

