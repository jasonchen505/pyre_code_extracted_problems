# Max Pooling 2D

**中文标题:** 二维最大池化

**难度:** Easy

**函数名:** `max_pool2d`

## 英文描述

Implement 2D max pooling from scratch.

Max pooling slides a window over the spatial dimensions of a feature map and takes the maximum value in each window.

**Signature:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`

**Parameters:**
- `x` — input tensor of shape (B, C, H, W)
- `kernel_size` — size of the pooling window (square)
- `stride` — step size between windows; defaults to `kernel_size` if `None`

**Returns:** tensor of shape (B, C, H_out, W_out) where
- `H_out = (H - kernel_size) // stride + 1`
- `W_out = (W - kernel_size) // stride + 1`

**Constraints:**
- Must match `torch.nn.functional.max_pool2d` numerically
- Do not call any `F.*` or `nn.functional.*` functions

## 中文描述

从零实现二维最大池化。

最大池化在特征图的空间维度上滑动窗口，取每个窗口内的最大值。

**签名:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`

**参数:**
- `x` — 输入张量，形状为 (B, C, H, W)
- `kernel_size` — 池化窗口大小（正方形）
- `stride` — 窗口步长；若为 `None` 则默认等于 `kernel_size`

**返回:** 形状为 (B, C, H_out, W_out) 的张量，其中
- `H_out = (H - kernel_size) // stride + 1`
- `W_out = (W - kernel_size) // stride + 1`

**约束:**
- 结果必须与 `torch.nn.functional.max_pool2d` 在数值上一致
- 不得调用任何 `F.*` 或 `nn.functional.*` 函数

## 提示

```
`x.unfold(2, k, stride).unfold(3, k, stride)` → `(B, C, H_out, W_out, k, k)` → `.flatten(-2).max(dim=-1).values`.
```

## 参考答案

```python
def max_pool2d(x, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    # unfold extracts sliding local blocks
    # after two unfolds: (B, C, H_out, W_out, kernel_size, kernel_size)
    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return patches.flatten(-2).max(dim=-1).values
```

## 示例代码

```python
x = torch.randn(1, 1, 4, 4)
out = max_pool2d(x, kernel_size=2, stride=2)
ref = F.max_pool2d(x, kernel_size=2, stride=2)
print("Output shape:", out.shape)
print("Matches F.max_pool2d:", torch.allclose(out, ref))
```

