# Sinusoidal Position Encoding

**中文标题:** 正弦位置编码

**难度:** Easy

**函数名:** `sinusoidal_pe`

## 英文描述

Implement the sinusoidal position encoding from 'Attention Is All You Need'.

Position encodings are added to token embeddings to give the model information about token positions. The original Transformer uses fixed sinusoidal functions.

**Signature:** `sinusoidal_pe(seq_len, d_model) -> Tensor`

**Parameters:**
- `seq_len` — number of positions
- `d_model` — embedding dimension (must be even)

**Returns:** position encoding tensor of shape `(seq_len, d_model)`

**Formula:**
- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

## 中文描述

实现《Attention Is All You Need》中的正弦位置编码。

位置编码加到词嵌入上，让模型感知 token 的位置信息。原始 Transformer 使用固定的正弦函数。

**签名:** `sinusoidal_pe(seq_len, d_model) -> Tensor`

**参数:**
- `seq_len` — 位置数量
- `d_model` — 嵌入维度（必须为偶数）

**返回:** 形状为 `(seq_len, d_model)` 的位置编码张量

**公式:**
- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

## 提示

```
1. `freqs = 1 / 10000^(2i/d_model)` for i in `range(d_model//2)`
2. `angles = pos[:, None] * freqs[None, :]` → shape `(seq_len, d_model//2)`
3. `pe[:, 0::2] = sin(angles)`, `pe[:, 1::2] = cos(angles)`
```

## 参考答案

```python
def sinusoidal_pe(seq_len, d_model):
    pos = torch.arange(seq_len).float().unsqueeze(1)
    dim = torch.arange(0, d_model, 2).float()
    freqs = 1.0 / (10000.0 ** (dim / d_model))
    angles = pos * freqs
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe
```

## 示例代码

```python
pe = sinusoidal_pe(10, 16)
print(pe.shape)
print(pe[:3, :4])
```

