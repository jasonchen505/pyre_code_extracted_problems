# GELU Activation

**中文标题:** GELU 激活函数

**难度:** Easy

**函数名:** `my_gelu`

## 英文描述

Implement the GELU activation function.

GELU (Gaussian Error Linear Unit) smoothly gates inputs based on their value, used in transformers like BERT and GPT.

**Signature:** `my_gelu(x) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape

**Returns:** element-wise GELU activation, same shape as input

**Constraints:**
- Exact formula: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- Must match `F.gelu` within 1e-4
- `gelu(0) = 0`

## 中文描述

实现 GELU 激活函数。

GELU（高斯误差线性单元）根据输入值平滑地进行门控，广泛用于 BERT 和 GPT 等 Transformer。

**签名:** `my_gelu(x) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量

**返回:** 逐元素 GELU 激活，形状与输入相同

**约束:**
- 精确公式：`x * 0.5 * (1 + erf(x / sqrt(2)))`
- 必须与 `F.gelu` 误差在 1e-4 以内
- `gelu(0) = 0`

## 提示

```
Exact: `0.5 * x * (1 + torch.erf(x / sqrt(2)))`
Approx: `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`
```

## 参考答案

```python
def my_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

## 示例代码

```python
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
```

