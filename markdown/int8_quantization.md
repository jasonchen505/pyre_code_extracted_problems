# INT8 Quantized Linear

**中文标题:** INT8 量化线性层

**难度:** Hard

**函数名:** `Int8Linear`

## 英文描述

Implement INT8 per-channel weight quantization for a linear layer.

Quantization converts float32 weights to int8 with per-channel scaling, reducing model size by 4x while preserving accuracy.

**Signature:** `Int8Linear(weight, bias=None)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** linear output with dequantized weights

**Constraints:**
- Per-channel scale: `abs(weight).max(dim=1) / 127`
- Quantize: `round(weight/scale).clamp(-128, 127).to(int8)`
- Store weight_int8 and scale as buffers, not parameters

## 中文描述

实现 INT8 逐通道权重量化线性层。

量化将 float32 权重转换为 int8 并使用逐通道缩放，将模型大小减少 4 倍同时保持精度。

**签名:** `Int8Linear(weight, bias=None)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** 使用反量化权重的线性输出

**约束:**
- 逐通道缩放：`abs(weight).max(dim=1) / 127`
- 量化：`round(weight/scale).clamp(-128, 127).to(int8)`
- weight_int8 和 scale 存储为 buffer 而非 parameter

## 提示

```
1. scale = abs(weight).max(dim=1) / 127  (per output channel, keepdim)
2. weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)
3. register both as buffers (not parameters)
4. forward: dequant = weight_int8.float() * scale → x @ dequant.T
```

## 参考答案

```python
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer('weight_int8',
            torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8))
        self.register_buffer('scale', scale)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out
```

## 示例代码

```python
w = torch.randn(8, 4)
q = Int8Linear(w)
print('Output:', q(torch.randn(2, 4)).shape)
print('Weight dtype:', q.weight_int8.dtype)
print('Compression: float32 -> int8 = 4x')
```

