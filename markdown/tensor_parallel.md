# Tensor Parallel MLP

**中文标题:** 张量并行 MLP

**难度:** Hard

**函数名:** `TensorParallelMLP`

## 英文描述

Implement a Megatron-style tensor parallel MLP.

Tensor parallelism splits the MLP weight matrices across devices. The first (up-projection) layer is column-parallel: each shard computes a slice of the hidden dimension. The second (down-projection) is row-parallel: each shard takes a slice of the input and produces the full output, then results are summed (all-reduce).

**Signature:** `TensorParallelMLP(d_model, d_ff, world_size)` (nn.Module)

**Parameters:**
- `d_model` — input/output dimension
- `d_ff` — hidden dimension (divisible by world_size)
- `world_size` — number of virtual shards

**Forward:** `forward(x) -> Tensor` where x is `(B, d_model)`

**Constraint:** Store sharded weights as `nn.ParameterList`. The forward pass must simulate column-parallel + row-parallel + all-reduce. Result must match a standard `d_model -> d_ff -> d_model` MLP with GELU activation.

## 中文描述

实现 Megatron 风格的张量并行 MLP。

张量并行将 MLP 权重矩阵分片到各设备。第一层（上投影）是列并行：每个分片计算隐藏维度的一个切片。第二层（下投影）是行并行：每个分片接收输入的一个切片并产生完整输出，然后求和（all-reduce）。

**签名:** `TensorParallelMLP(d_model, d_ff, world_size)`（nn.Module）

**参数:**
- `d_model` — 输入/输出维度
- `d_ff` — 隐藏维度（可被 world_size 整除）
- `world_size` — 虚拟分片数

**前向:** `forward(x) -> Tensor`，x 形状 `(B, d_model)`

**约束:** 将分片权重存为 `nn.ParameterList`。前向传播必须模拟列并行 + 行并行 + all-reduce。结果必须与带 GELU 激活的标准 `d_model -> d_ff -> d_model` MLP 一致。

## 提示

```
1. W1: column-split → w1_shards[i] shape `(d_model, d_ff/world_size)`
2. W2: row-split → w2_shards[i] shape `(d_ff/world_size, d_model)`
3. for each shard i: h_i = gelu(x @ w1_i) → partial_i = h_i @ w2_i
4. all-reduce = sum(partial_i for all i)
```

## 参考答案

```python
class TensorParallelMLP(nn.Module):
    def __init__(self, d_model, d_ff, world_size):
        super().__init__()
        self.world_size = world_size
        shard_size = d_ff // world_size
        # Column-parallel: W1 shards (d_model, shard_size) each
        self.w1_shards = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, shard_size) * (2 / d_model) ** 0.5)
            for _ in range(world_size)
        ])
        # Row-parallel: W2 shards (shard_size, d_model) each
        self.w2_shards = nn.ParameterList([
            nn.Parameter(torch.randn(shard_size, d_model) * (2 / d_ff) ** 0.5)
            for _ in range(world_size)
        ])

    def forward(self, x):
        # Column-parallel forward + row-parallel + all-reduce (sum)
        output = None
        for w1, w2 in zip(self.w1_shards, self.w2_shards):
            z = x @ w1
            h = z * 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))  # GELU
            partial = h @ w2                        # (B, d_model)
            output = partial if output is None else output + partial
        return output
```

## 示例代码

```python
torch.manual_seed(42)
d_model, d_ff, world_size = 64, 256, 4
x = torch.randn(2, d_model)

tp_mlp = TensorParallelMLP(d_model, d_ff, world_size)
out = tp_mlp(x)
print("Output shape:", out.shape)  # expect (2, 64)

w1_full = torch.cat([p.data for p in tp_mlp.w1_shards], dim=1)  # (d_model, d_ff)
w2_full = torch.cat([p.data for p in tp_mlp.w2_shards], dim=0)  # (d_ff, d_model)
ref = torch.nn.functional.gelu(x @ w1_full) @ w2_full

print("Max diff vs reference:", (out - ref).abs().max().item())  # expect ~0
```

