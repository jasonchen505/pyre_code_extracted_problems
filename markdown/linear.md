# Simple Linear Layer

**中文标题:** 简单线性层

**难度:** Medium

**函数名:** `SimpleLinear`

## 英文描述

Implement a simple linear (fully connected) layer from scratch.

A linear layer computes `y = x @ W^T + b` with learnable weight and bias tensors.

**Signature:** `SimpleLinear(in_features, out_features)` (class)

**Method:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** output tensor (*, out_features)

**Constraints:**
- Weight shape: (out_features, in_features) with Kaiming scaling
- Bias shape: (out_features,) initialized to zeros
- Both must have `requires_grad=True`

## 中文描述

从零实现简单线性（全连接）层。

线性层使用可学习的权重和偏置张量计算 `y = x @ W^T + b`。

**签名:** `SimpleLinear(in_features, out_features)`（类）

**方法:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** 输出张量 (*, out_features)

**约束:**
- 权重形状：(out_features, in_features)，使用 Kaiming 缩放
- 偏置形状：(out_features,)，初始化为零
- 两者都必须 `requires_grad=True`

## 提示

```
`weight` shape `(out, in)`, init `randn * 1/sqrt(in_features)`
`bias` shape `(out,)`, init zeros
Forward: `x @ weight.T + bias`
```

## 参考答案

```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```

## 示例代码

```python
layer = SimpleLinear(8, 4)
print("W shape:", layer.weight.shape)
print("b shape:", layer.bias.shape)
x = torch.randn(2, 8)
print("Output shape:", layer.forward(x).shape)
```

