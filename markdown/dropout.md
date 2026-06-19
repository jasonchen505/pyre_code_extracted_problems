# Implement Dropout

**中文标题:** 实现 Dropout

**难度:** Easy

**函数名:** `MyDropout`

## 英文描述

Implement dropout as an nn.Module.

Dropout randomly zeroes elements during training and scales survivors by `1/(1-p)` to maintain expected values. During eval, it is an identity.

**Signature:** `MyDropout(p=0.5)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor of any shape

**Returns:** tensor with dropout applied (training) or unchanged (eval)

**Constraints:**
- Training: zero with probability p, scale by `1/(1-p)`
- Eval: return input unchanged

## 中文描述

实现 Dropout（nn.Module）。

Dropout 在训练时以概率 p 随机将元素置零，并将存活元素缩放 `1/(1-p)` 以保持期望值不变。推理时为恒等映射。

**签名:** `MyDropout(p=0.5)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 任意形状的输入张量

**返回:** 应用 dropout 后的张量（训练）或原始输入（推理）

**约束:**
- 训练模式：以概率 p 置零，缩放 `1/(1-p)`
- 推理模式：返回原始输入

## 提示

```
Train: `mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`
Eval: return `x` unchanged
```

## 参考答案

```python
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)
```

## 示例代码

```python
d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))
```

