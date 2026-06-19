# SwiGLU Activation

**中文标题:** SwiGLU 激活函数

**难度:** Medium

**函数名:** `swiglu`

## 英文描述

Implement the SwiGLU gated activation function.

SwiGLU is used in LLaMA, PaLM, and other modern LLMs as the MLP activation. It gates one linear projection with a Swish-activated version of another.

**Signature:** `swiglu(x, W1, W2, Wgate) -> Tensor`

**Parameters:**
- `x` — input tensor `(B, d_model)`
- `W1` — weight matrix `(d_model, d_ff)`
- `W2` — output projection `(d_ff, d_model)`
- `Wgate` — gate weight `(d_model, d_ff)`

**Returns:** output tensor `(B, d_model)`

**Formula:** `hidden = (x @ W1) * swish(x @ Wgate)`, output = `hidden @ W2`

**Swish:** `swish(z) = z * sigmoid(z)`

## 中文描述

实现 SwiGLU 门控激活函数。

SwiGLU 被 LLaMA、PaLM 等现代大语言模型用作 MLP 激活函数，用一个线性投影的 Swish 激活版本对另一个进行门控。

**签名:** `swiglu(x, W1, W2, Wgate) -> Tensor`

**参数:**
- `x` — 输入张量 `(B, d_model)`
- `W1` — 权重矩阵 `(d_model, d_ff)`
- `W2` — 输出投影 `(d_ff, d_model)`
- `Wgate` — 门控权重 `(d_model, d_ff)`

**返回:** 输出张量 `(B, d_model)`

**公式:** 隐藏状态 = `(x @ W1) * swish(x @ Wgate)`，输出 = `隐藏状态 @ W2`

**Swish:** `swish(z) = z * sigmoid(z)`

## 提示

```
1. `gate = (x @ Wgate) * sigmoid(x @ Wgate)`  ← swish
2. `hidden = (x @ W1) * gate`
3. `return hidden @ W2`
```

## 参考答案

```python
def swiglu(x, W1, W2, Wgate):
    h = x @ W1
    gate = x @ Wgate
    swish_gate = gate * torch.sigmoid(gate)
    return (h * swish_gate) @ W2
```

## 示例代码

```python
torch.manual_seed(0)
B, D, H = 2, 8, 16
x = torch.randn(B, D)
W1 = torch.randn(D, H)
W2 = torch.randn(H, D)
Wgate = torch.randn(D, H)

out = swiglu(x, W1, W2, Wgate)
print("Output shape:", out.shape)

gate = x @ Wgate
swish_gate = gate * torch.sigmoid(gate)
print("Gate (swish) sample values:", swish_gate[0, :4])
```

