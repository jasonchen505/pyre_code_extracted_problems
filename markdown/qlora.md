# QLoRA

**中文标题:** QLoRA

**难度:** Hard

**函数名:** `QLoRALinear`

## 英文描述

Implement QLoRA: a quantized linear layer with LoRA adapters.

QLoRA stores the base weight in int8 (simulating 4-bit quantization) with a per-row fp32 scale, while LoRA adapters remain in full precision for efficient fine-tuning.

**Signature:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Parameters stored:**
- `quantized_weight`: (out_features, in_features), dtype=int8, requires_grad=False
- `scale`: (out_features, 1), fp32 per-row scale
- `lora_A`: (in_features, rank), initialized ~ N(0, 0.01²)
- `lora_B`: (rank, out_features), initialized to zeros

**Forward:**
1. Dequantize: `W_fp = quantized_weight.float() * scale`
2. Base output: `x @ W_fp.T`
3. LoRA delta: `x @ lora_A @ lora_B * (alpha / rank)`
4. Return base + delta

**Helper:** `set_weight(W_fp32)` quantizes a float weight per-row:
- `scale = W.abs().max(dim=1, keepdim=True).values / 127`
- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`

## 中文描述

实现 QLoRA：带 LoRA 适配器的量化线性层。

QLoRA 将基础权重以 int8 格式存储（模拟 4-bit 量化），配合每行 fp32 缩放因子，同时 LoRA 适配器保持全精度，实现高效微调。

**签名:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）

**存储的参数:**
- `quantized_weight`：(out_features, in_features)，dtype=int8，requires_grad=False
- `scale`：(out_features, 1)，fp32 每行缩放因子
- `lora_A`：(in_features, rank)，初始化 ~ N(0, 0.01²)
- `lora_B`：(rank, out_features)，初始化为零

**前向传播:**
1. 反量化：`W_fp = quantized_weight.float() * scale`
2. 基础输出：`x @ W_fp.T`
3. LoRA 增量：`x @ lora_A @ lora_B * (alpha / rank)`
4. 返回 base + delta

**辅助方法:** `set_weight(W_fp32)` 按行量化浮点权重：
- `scale = W.abs().max(dim=1, keepdim=True).values / 127`
- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`

## 提示

```
set_weight: `scale = W.abs().max(dim=1,keepdim=True).values.clamp(1e-8)/127`
           `q = (W/scale).round().clamp(-127,127).to(int8)`
Forward: `W_fp = q.float()*scale` → `x@W_fp.T + x@lora_A@lora_B*(alpha/rank)`
```

## 参考答案

```python
class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.quantized_weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def set_weight(self, W_fp32):
        scale = W_fp32.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
        q = (W_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
        self.quantized_weight.data.copy_(q)
        self.scale.data.copy_(scale)

    def forward(self, x):
        W_fp = self.quantized_weight.float() * self.scale
        base = x @ W_fp.T
        delta = x @ self.lora_A @ self.lora_B * (self.alpha / self.rank)
        return base + delta
```

## 示例代码

```python
torch.manual_seed(0)
in_f, out_f, rank = 64, 32, 4
layer = QLoRALinear(in_f, out_f, rank, alpha=2.0)

W_ref = torch.randn(out_f, in_f)
layer.set_weight(W_ref)

x = torch.randn(8, in_f)
y_qlora = layer(x)
y_ref = x @ W_ref.T  # full-precision baseline (no LoRA delta)

print("Output shape:", y_qlora.shape)          # (8, 32)
print("Max abs error vs fp32:", (y_qlora - y_ref).abs().max().item())
```

