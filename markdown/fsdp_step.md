# FSDP Training Step

**中文标题:** FSDP 训练步骤

**难度:** Hard

**函数名:** `fsdp_step`

## 英文描述

Implement a simplified FSDP (Fully Sharded Data Parallel) training step.

FSDP shards model parameters across workers to reduce per-device memory. Before the forward pass, each worker all-gathers the full parameters. After the backward pass, gradients are reduce-scattered so each worker only holds its shard's gradient.

**Signature:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`

**Parameters:**
- `param_shards` — list of `world_size` parameter shards (each a flat 1D tensor)
- `grad_fn` — callable that takes the full (unsharded) parameter tensor and returns the gradient w.r.t. it
- `world_size` — number of virtual workers

**Returns:** updated list of parameter shards (after one gradient step with lr=0.01)

**Steps:**
1. All-gather: concatenate shards to get full parameter
2. Compute gradient via grad_fn(full_param)
3. Reduce-scatter: split gradient into world_size chunks, each worker keeps its chunk
4. Update each shard: shard -= 0.01 * grad_shard

## 中文描述

实现简化版 FSDP（全分片数据并行）训练步骤。

FSDP 将模型参数分片到各 worker 以减少单设备内存。前向传播前，每个 worker all-gather 完整参数。反向传播后，梯度 reduce-scatter，每个 worker 只保留自己分片的梯度。

**签名:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`

**参数:**
- `param_shards` — `world_size` 个参数分片的列表（每个是扁平 1D 张量）
- `grad_fn` — 接收完整（未分片）参数张量，返回对应梯度的可调用对象
- `world_size` — 虚拟 worker 数量

**返回:** 更新后的参数分片列表（一步梯度更新，lr=0.01）

**步骤:**
1. All-gather：拼接分片得到完整参数
2. 通过 grad_fn(full_param) 计算梯度
3. Reduce-scatter：将梯度分成 world_size 块，每个 worker 保留自己的块
4. 更新每个分片：shard -= 0.01 * grad_shard

## 提示

```
1. all-gather: full_param = torch.cat(param_shards)
2. grad = grad_fn(full_param)
3. reduce-scatter: grad_shards = list(grad.chunk(world_size))
4. new_shards[i] = param_shards[i] - 0.01 * grad_shards[i]
```

## 参考答案

```python
def fsdp_step(param_shards, grad_fn, world_size):
    # 1. All-gather: reconstruct full parameter
    full_param = torch.cat(param_shards)

    # 2. Compute gradient
    grad = grad_fn(full_param)

    # 3. Reduce-scatter: each worker gets its gradient shard
    grad_shards = list(grad.chunk(world_size))

    # 4. Update each shard with lr=0.01
    new_shards = [
        param_shards[i] - 0.01 * grad_shards[i]
        for i in range(world_size)
    ]
    return new_shards
```

## 示例代码

```python
torch.manual_seed(0)
world_size = 4
shard_size = 8

param_shards = [torch.randn(shard_size) for _ in range(world_size)]

grad_fn = lambda p: 2.0 * p

new_shards = fsdp_step(param_shards, grad_fn, world_size)
fsdp_result = torch.cat(new_shards)

full_param = torch.cat(param_shards)
ref_result = full_param - 0.01 * grad_fn(full_param)

print("FSDP result shape:", fsdp_result.shape)  # expect (32,)
print("Max diff vs SGD reference:", (fsdp_result - ref_result).abs().max().item())  # expect ~0
```

