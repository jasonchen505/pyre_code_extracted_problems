# Implement LayerNorm

**中文标题:** 实现 LayerNorm

**难度:** Medium

**函数名:** `my_layer_norm`

## 英文描述

Implement Layer Normalization.

LayerNorm normalizes each sample across the feature dimension, stabilizing training without dependence on batch size.

**Signature:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`

**Parameters:**
- `x` — input tensor (..., D)
- `gamma` — scale parameter (D,)
- `beta` — shift parameter (D,)
- `eps` — epsilon for numerical stability

**Returns:** normalized tensor, same shape as x

**Constraints:**
- Normalize over the last dimension
- Use `unbiased=False` for variance
- Must match `F.layer_norm`

## 中文描述

实现层归一化。

LayerNorm 对每个样本沿特征维度进行归一化，不依赖批大小即可稳定训练。

**签名:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`

**参数:**
- `x` — 输入张量 (..., D)
- `gamma` — 缩放参数 (D,)
- `beta` — 偏移参数 (D,)
- `eps` — 数值稳定性的 epsilon

**返回:** 归一化后的张量，形状与 x 相同

**约束:**
- 沿最后一个维度归一化
- 方差使用 `unbiased=False`
- 必须与 `F.layer_norm` 一致

## 提示

```
1. `mean = x.mean(dim=-1, keepdim=True)`
2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`
3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`
```

## 参考答案

```python
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

## 示例代码

```python
x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))
```

