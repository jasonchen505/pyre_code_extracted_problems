# Linear Regression

**中文标题:** 线性回归

**难度:** Medium

**函数名:** `LinearRegression`

## 英文描述

Implement linear regression three ways: closed-form, gradient descent, and nn.Linear.

This task covers the fundamental approaches to fitting a linear model, from the normal equation to autograd-based training.

**Signature:** `LinearRegression()` (class)

**Methods:**
- `closed_form(X, y) -> (w, b)` — solve via normal equation
- `gradient_descent(X, y, lr, steps) -> (w, b)` — manual gradient updates
- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd training loop

**Constraints:**
- All three methods should converge to similar weights
- Closed-form should not use autograd
- X is (N, D), y is (N,), returns w (D,) and b (scalar)

## 中文描述

用三种方式实现线性回归：闭式解、梯度下降和 nn.Linear。

本题涵盖拟合线性模型的基本方法，从正规方程到基于 autograd 的训练。

**签名:** `LinearRegression()`（类）

**方法:**
- `closed_form(X, y) -> (w, b)` — 通过正规方程求解
- `gradient_descent(X, y, lr, steps) -> (w, b)` — 手动梯度更新
- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd 训练循环

**约束:**
- 三种方法应收敛到相近的权重
- 闭式解不应使用 autograd
- X 为 (N, D)，y 为 (N,)，返回 w (D,) 和 b（标量）

## 提示

```
closed_form: augment X with ones col → `lstsq(X_aug, y)` → split `w, b`
gradient_descent: `grad_w = (2/N) * X.T @ (X@w+b - y)`, `w -= lr*grad_w`
nn_linear: `nn.Linear(D,1)` + `MSELoss` + `optimizer.step()` loop
```

## 参考答案

```python
class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation via augmented matrix."""
        N, D = X.shape
        # Augment X with ones column for bias
        X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, D+1)
        # Solve (X^T X) theta = X^T y
        theta = torch.linalg.lstsq(X_aug, y).solution      # (D+1,)
        w = theta[:D]
        b = theta[D]
        return w.detach(), b.detach()

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient computation — no autograd."""
        N, D = X.shape
        w = torch.zeros(D)
        b = torch.tensor(0.0)

        for _ in range(steps):
            pred = X @ w + b          # (N,)
            error = pred - y           # (N,)
            grad_w = (2.0 / N) * (X.T @ error)  # (D,)
            grad_b = (2.0 / N) * error.sum()     # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b

        return w, b

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """PyTorch nn.Linear with autograd training loop."""
        N, D = X.shape
        layer = nn.Linear(D, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(steps):
            optimizer.zero_grad()
            pred = layer(X).squeeze(-1)  # (N,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        w = layer.weight.data.squeeze(0)  # (D,)
        b = layer.bias.data.squeeze(0)    # scalar ()
        return w, b
```

## 示例代码

```python
torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0

model = LinearRegression()
for name, method in [("Closed-form", model.closed_form),
                      ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                      ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
    w, b = method(X, y)
    print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")
```

