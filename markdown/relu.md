# Implement ReLU

**中文标题:** 实现 ReLU

**难度:** Easy

**函数名:** `relu`

## 英文描述

Implement the ReLU activation function.

ReLU (Rectified Linear Unit) outputs the input directly if positive, otherwise zero. It is the most widely used activation in deep learning.

**Signature:** `relu(x) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape

**Returns:** element-wise ReLU activation, same shape as input

**Constraints:**
- `relu(x) = max(0, x)` element-wise
- Must support gradient flow
- Must be efficient on large tensors

## 中文描述

实现 ReLU 激活函数。

ReLU（修正线性单元）在输入为正时直接输出，否则输出零，是深度学习中最广泛使用的激活函数。

**签名:** `relu(x) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量

**返回:** 逐元素 ReLU 激活，形状与输入相同

**约束:**
- `relu(x) = max(0, x)` 逐元素
- 必须支持梯度流
- 在大张量上必须高效

## 提示

```
`relu(x) = max(0, x)` element-wise → `x * (x > 0)`
```

## 参考答案

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()
```

## 示例代码

```python
x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
```

