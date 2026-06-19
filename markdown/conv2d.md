# 2D Convolution

**中文标题:** 二维卷积

**难度:** Medium

**函数名:** `my_conv2d`

## 英文描述

Implement 2D convolution from scratch.

Convolution slides a kernel over a 2D input, computing weighted sums at each position. It is the fundamental operation in CNNs.

**Signature:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`

**Parameters:**
- `x` — input tensor (B, C_in, H, W)
- `weight` — kernel tensor (C_out, C_in, kH, kW)
- `bias` — optional bias (C_out,)
- `stride`, `padding` — integer stride and zero-padding

**Returns:** convolved output tensor

**Constraints:**
- Must match `F.conv2d` numerically
- Support stride and padding parameters

## 中文描述

从零实现二维卷积。

卷积将卷积核在二维输入上滑动，在每个位置计算加权和，是 CNN 的基本操作。

**签名:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`

**参数:**
- `x` — 输入张量 (B, C_in, H, W)
- `weight` — 卷积核张量 (C_out, C_in, kH, kW)
- `bias` — 可选偏置 (C_out,)
- `stride`, `padding` — 整数步幅和零填充

**返回:** 卷积输出张量

**约束:**
- 必须与 `F.conv2d` 数值一致
- 支持 stride 和 padding 参数

## 提示

```
1. Zero-pad input manually (no F.pad)
2. `x.unfold(2, kH, stride).unfold(3, kW, stride)` → patches `(B, C, H_out, W_out, kH, kW)`
3. `(patches * weight).sum((-3,-2,-1)) + bias`

```

## 参考答案

```python
def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        B, C, H, W = x.shape
        x_pad = torch.zeros(B, C, H + 2*padding, W + 2*padding, dtype=x.dtype, device=x.device)
        x_pad[:, :, padding:padding+H, padding:padding+W] = x
        x = x_pad
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
    out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
```

## 示例代码

```python
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print('Output:', my_conv2d(x, w).shape)
print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
```

