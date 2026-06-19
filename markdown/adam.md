# Adam Optimizer

**中文标题:** Adam 优化器

**难度:** Medium

**函数名:** `MyAdam`

## 英文描述

Implement the Adam optimizer from scratch.

Adam combines momentum (1st moment) and RMSProp (2nd moment) with bias correction for adaptive per-parameter learning rates.

**Signature:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`

**Methods:**
- `step()` — update parameters using bias-corrected moments
- `zero_grad()` — zero all parameter gradients

**Constraints:**
- Must match `torch.optim.Adam` numerically
- Bias correction: `m_hat = m / (1 - beta1^t)`

## 中文描述

从零实现 Adam 优化器。

Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），并通过偏差校正实现自适应的逐参数学习率。

**签名:** `MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`

**方法:**
- `step()` — 使用偏差校正后的矩更新参数
- `zero_grad()` — 将所有参数梯度清零

**约束:**
- 必须与 `torch.optim.Adam` 数值一致
- 偏差校正: `m_hat = m / (1 - beta1^t)`

## 提示

```
1. m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²
2. Bias-correct: m̂ = m/(1-β1ᵗ),  v̂ = v/(1-β2ᵗ)
3. p -= lr · m̂ / (√v̂ + ε)
```

## 参考答案

```python
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

## 示例代码

```python
torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = MyAdam([w], lr=0.01)
for i in range(5):
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'Step {i}: loss={loss.item():.4f}')
```

