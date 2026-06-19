# Kaiming Initialization

**中文标题:** Kaiming 初始化

**难度:** Easy

**函数名:** `kaiming_init`

## 英文描述

Implement Kaiming (He) weight initialization.

Kaiming initialization sets weight variance based on fan-in to preserve signal magnitude through ReLU networks, preventing vanishing/exploding activations.

**Signature:** `kaiming_init(weight) -> Tensor`

**Parameters:**
- `weight` — tensor to initialize in-place (out_features, in_features)

**Returns:** the same tensor (in-place operation)

**Constraints:**
- `std = sqrt(2 / fan_in)` where `fan_in = weight.shape[1]`
- Fill with `normal(0, std)`
- Smaller fan_in should give larger std

## 中文描述

实现 Kaiming（He）权重初始化。

Kaiming 初始化根据 fan-in 设置权重方差，以在 ReLU 网络中保持信号幅度，防止激活值消失或爆炸。

**签名:** `kaiming_init(weight) -> Tensor`

**参数:**
- `weight` — 需要原地初始化的张量 (out_features, in_features)

**返回:** 同一张量（原地操作）

**约束:**
- `std = sqrt(2 / fan_in)`，其中 `fan_in = weight.shape[1]`
- 用 `normal(0, std)` 填充
- 更小的 fan_in 应产生更大的 std

## 提示

```
`fan_in = weight.shape[1]` → `std = sqrt(2 / fan_in)`
`weight.normal_(0, std)` in-place → return `weight`
```

## 参考答案

```python
def kaiming_init(weight):
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0, std)
    return weight
```

## 示例代码

```python
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')
```

