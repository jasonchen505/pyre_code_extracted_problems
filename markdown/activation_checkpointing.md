# Activation Checkpointing

**中文标题:** 激活检查点

**难度:** Medium

**函数名:** `checkpoint_sequential`

## 英文描述

Implement gradient checkpointing for a sequence of functions.

Activation checkpointing trades compute for memory: instead of storing all intermediate activations during the forward pass, they are recomputed on-the-fly during backpropagation.

**Signature:** `checkpoint_sequential(fns, x) -> Tensor`

**Parameters:**
- `fns` — list of callables, each mapping Tensor -> Tensor
- `x` — input tensor

**Returns:** output tensor after applying all functions sequentially

**Constraints:**
- Intermediate activations must NOT be stored during the forward pass
- Gradients must flow back to `x`
- Output must be numerically identical to naive sequential application
- Use `torch.utils.checkpoint.checkpoint` (allowed — it is not `F.*` or `nn.functional.*`)

## 中文描述

为一系列函数实现梯度检查点。

激活检查点以计算换内存：在前向传播期间不存储所有中间激活，而是在反向传播时按需重新计算。

**签名:** `checkpoint_sequential(fns, x) -> Tensor`

**参数:**
- `fns` — 可调用对象列表，每个接受 Tensor 并返回 Tensor
- `x` — 输入张量

**返回:** 依次应用所有函数后的输出张量

**约束:**
- 前向传播期间不得存储中间激活
- 梯度必须能流回 `x`
- 输出必须与朴素顺序应用在数值上完全一致
- 使用 `torch.utils.checkpoint.checkpoint`（允许——它不是 `F.*` 或 `nn.functional.*`）

## 提示

```
import torch.utils.checkpoint as cp
Loop over fns: x = cp.checkpoint(fn, x, use_reentrant=False)
Activations are recomputed on backward instead of stored.
```

## 参考答案

```python
import torch.utils.checkpoint as cp

def checkpoint_sequential(fns, x):
    for fn in fns:
        x = cp.checkpoint(fn, x, use_reentrant=False)
    return x
```

## 示例代码

```python
torch.manual_seed(0)
layers = [torch.nn.Linear(16, 16) for _ in range(4)]
x = torch.randn(4, 16, requires_grad=True)

out_cp = checkpoint_sequential(layers, x)

x2 = x.detach().requires_grad_(True)
out_naive = x2
for layer in layers:
    out_naive = layer(out_naive)

print("Outputs match:", torch.allclose(out_cp, out_naive, atol=1e-5))

out_cp.sum().backward()
print("Gradient flows (x.grad is not None):", x.grad is not None)
```

