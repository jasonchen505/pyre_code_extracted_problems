# Gradient Accumulation

**中文标题:** 梯度累积

**难度:** Easy

**函数名:** `accumulated_step`

## 英文描述

Implement gradient accumulation over micro-batches.

Gradient accumulation simulates a large batch by accumulating gradients from multiple smaller batches before a single optimizer step.

**Signature:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`

**Parameters:**
- `model` — nn.Module
- `optimizer` — torch optimizer
- `loss_fn` — loss function
- `micro_batches` — list of (x, y) tuples

**Returns:** total loss as a float

**Constraints:**
- Scale each micro-batch loss by `1/n` before backward
- Must match a single full-batch update numerically

## 中文描述

实现微批次梯度累积。

梯度累积通过在多个小批次上累积梯度后执行一次优化器步骤，模拟大批次训练。

**签名:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`

**参数:**
- `model` — nn.Module
- `optimizer` — torch 优化器
- `loss_fn` — 损失函数
- `micro_batches` — (x, y) 元组列表

**返回:** 总损失（浮点数）

**约束:**
- 每个微批次损失在反向传播前除以 `n`
- 必须与单次全批次更新数值一致

## 提示

```
1. optimizer.zero_grad() once
2. for each micro-batch: forward → loss/n → backward
3. optimizer.step()
   Dividing by n ensures accumulated grads match a single full-batch update.
```

## 参考答案

```python
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss
```

## 示例代码

```python
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)
```

