# Implement Softmax

**中文标题:** 实现 Softmax

**难度:** Easy

**函数名:** `my_softmax`

## 英文描述

Implement the softmax function.

Softmax converts raw logits into a probability distribution by exponentiating and normalizing, used in classification and attention.

**Signature:** `my_softmax(x, dim=-1) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape
- `dim` — dimension along which to compute softmax

**Returns:** probability tensor (sums to 1 along dim), same shape as input

**Constraints:**
- Subtract max for numerical stability before exp
- Must handle large values without NaN/Inf

## 中文描述

实现 softmax 函数。

Softmax 通过指数化和归一化将原始 logits 转换为概率分布，用于分类和注意力机制。

**签名:** `my_softmax(x, dim=-1) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量
- `dim` — 计算 softmax 的维度

**返回:** 概率张量（沿 dim 求和为 1），形状与输入相同

**约束:**
- 在 exp 之前减去最大值以保证数值稳定
- 必须处理大值而不产生 NaN/Inf

## 提示

```
1. `x_max = x.max(dim=dim, keepdim=True).values`
2. `e_x = exp(x - x_max)`
3. `return e_x / e_x.sum(dim=dim, keepdim=True)`
```

## 参考答案

```python
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

## 示例代码

```python
x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())
print("Ref:   ", torch.softmax(x, dim=-1))
```

