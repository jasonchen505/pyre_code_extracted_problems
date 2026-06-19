# Depthwise Separable Convolution

**中文标题:** 深度可分离卷积

**难度:** Medium

**函数名:** `depthwise_separable_conv`

## 英文描述

Implement MobileNet-style depthwise separable convolution from primitives.

Depthwise separable convolution factorizes a standard convolution into two steps: a depthwise conv (one filter per input channel) followed by a pointwise 1x1 conv (to mix channels). This dramatically reduces parameter count and FLOPs.

**Signature:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`

**Parameters:**
- `x` — input tensor (B, C_in, H, W)
- `dw_weight` — depthwise filter (C_in, 1, kH, kW)
- `pw_weight` — pointwise filter (C_out, C_in, 1, 1)

**Returns:** output tensor (B, C_out, H-kH+1, W-kW+1)

**Constraints:**
- No padding, stride=1
- Must NOT use `F.conv2d` — implement both steps using `unfold` and `einsum`/matmul
- Depthwise step: each channel c is convolved only with `dw_weight[c, 0]`
- Pointwise step: 1x1 conv mixes channels via matrix multiply

## 中文描述

从基本原语实现 MobileNet 风格的深度可分离卷积。

深度可分离卷积将标准卷积分解为两步：深度卷积（每个输入通道一个滤波器）和逐点 1x1 卷积（混合通道）。这大幅减少了参数量和计算量。

**签名:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`

**参数:**
- `x` — 输入张量 (B, C_in, H, W)
- `dw_weight` — 深度滤波器 (C_in, 1, kH, kW)
- `pw_weight` — 逐点滤波器 (C_out, C_in, 1, 1)

**返回:** 输出张量 (B, C_out, H-kH+1, W-kW+1)

**约束:**
- 无填充，步长=1
- 不得使用 `F.conv2d`——使用 `unfold` 和 `einsum`/matmul 实现两步
- 深度步骤：通道 c 仅与 `dw_weight[c, 0]` 卷积
- 逐点步骤：1x1 卷积通过矩阵乘法混合通道

## 提示

```
Depthwise: `x.unfold(2,kH,1).unfold(3,kW,1)` → `(B,C,H_out,W_out,kH,kW)` → multiply `dw_weight` → sum last 2 dims
Pointwise: `torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:,: ,0,0])`

```

## 参考答案

```python
def depthwise_separable_conv(x, dw_weight, pw_weight):
    B, C_in, H, W = x.shape
    kH, kW = dw_weight.shape[2], dw_weight.shape[3]
    H_out, W_out = H - kH + 1, W - kW + 1
    # Depthwise: unfold spatial dims to extract patches
    patches = x.unfold(2, kH, 1).unfold(3, kW, 1)  # (B, C_in, H_out, W_out, kH, kW)
    dw_out = (patches * dw_weight[:, 0].view(1, C_in, 1, 1, kH, kW)).sum(dim=(-2, -1))  # (B, C_in, H_out, W_out)
    # Pointwise: 1x1 conv = channel-wise linear combination
    out = torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:, :, 0, 0])
    return out
```

## 示例代码

```python
torch.manual_seed(0)
B, C_in, H, W = 2, 4, 8, 8
C_out, kH, kW = 8, 3, 3

x         = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)   # one kernel per input channel
pw_weight = torch.randn(C_out, C_in, 1, 1)  # 1x1 conv

out = depthwise_separable_conv(x, dw_weight, pw_weight)
print("Output shape:", out.shape)  # (2, 8, 6, 6)

patches = x.unfold(2, kH, 1).unfold(3, kW, 1)
dw_ch0 = (patches[:, 0:1] * dw_weight[0:1, 0].view(1, 1, 1, 1, kH, kW)).sum(dim=(-2, -1))
dw_ch1 = (patches[:, 1:2] * dw_weight[1:2, 0].view(1, 1, 1, 1, kH, kW)).sum(dim=(-2, -1))
print("DW ch0 and ch1 are independent (cross-correlation ~0):",
      (dw_ch0 * dw_ch1).mean().abs().item() < 1.0)
```

