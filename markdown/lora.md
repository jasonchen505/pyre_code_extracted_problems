# LoRA (Low-Rank Adaptation)

**中文标题:** LoRA（低秩适配）

**难度:** Medium

**函数名:** `LoRALinear`

## 英文描述

Implement LoRA (Low-Rank Adaptation) for a linear layer.

LoRA freezes the base weights and adds trainable low-rank matrices A and B, enabling efficient fine-tuning with far fewer parameters.

**Signature:** `LoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`

**Constraints:**
- Base `nn.Linear` weights must be frozen (requires_grad=False)
- `lora_A`: (rank, in_features), `lora_B`: (out_features, rank) initialized to zeros
- Only LoRA params should receive gradients

## 中文描述

实现线性层的 LoRA（低秩适配）。

LoRA 冻结基础权重并添加可训练的低秩矩阵 A 和 B，以极少的参数实现高效微调。

**签名:** `LoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** `linear(x) + (x @ A^T @ B^T) * (alpha/rank)`

**约束:**
- 基础 `nn.Linear` 权重必须冻结（requires_grad=False）
- `lora_A`：(rank, in_features)，`lora_B`：(out_features, rank) 初始化为零
- 只有 LoRA 参数应接收梯度

## 提示

```
1. Freeze `linear.weight/bias` with `requires_grad_(False)`
2. `lora_A`: `(rank, in)`, `lora_B`: `(out, rank)` init zeros
3. Forward: `linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`
```

## 参考答案

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

## 示例代码

```python
layer = LoRALinear(16, 8, rank=4)
x = torch.randn(2, 16)
print('Output:', layer(x).shape)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f'Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)')
```

