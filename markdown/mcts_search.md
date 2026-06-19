# MCTS for Reasoning

**中文标题:** 推理蒙特卡洛树搜索

**难度:** Hard

**函数名:** `mcts_step`

## 英文描述

Implement one step of Monte Carlo Tree Search (MCTS) — the core algorithm behind o1/AlphaProof-style LLM reasoning.

**Signature:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`

**Parameters:**
- `values` — current value estimates for N child nodes (N,)
- `visit_counts` — visit counts for each child (N,)
- `parent_visits` — total visits to the parent node (int or scalar)
- `c_puct` — exploration constant (default 1.414)

**Returns:** `(selected_idx, updated_values, updated_visits)`

**Algorithm:**
1. **Selection** — compute UCB1 scores and pick `argmax`:
   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`
2. **Rollout** — simulate a random value in [0, 1] for the selected node
3. **Backpropagation** — update with running mean:
   `Q_new = (Q_old * n + rollout_value) / (n + 1)` where `n` is the old visit count
4. Increment `visit_counts[selected_idx]` by 1

## 中文描述

实现蒙特卡洛树搜索（MCTS）的单步操作——o1/AlphaProof 风格 LLM 推理的核心算法。

**签名:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`

**参数:**
- `values` — N 个子节点的当前价值估计 (N,)
- `visit_counts` — 每个子节点的访问次数 (N,)
- `parent_visits` — 父节点的总访问次数（int 或标量）
- `c_puct` — 探索常数（默认 1.414）

**返回:** `(selected_idx, updated_values, updated_visits)`

**算法:**
1. **选择** — 计算 UCB1 分数并取 `argmax`：
   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`
2. **模拟** — 为选中节点模拟 [0, 1] 内的随机价值
3. **反向传播** — 用滑动均值更新：
   `Q_new = (Q_old * n + rollout_value) / (n + 1)`，其中 `n` 为旧访问次数
4. 将 `visit_counts[selected_idx]` 加 1

## 提示

```
1. UCB(i) = values[i] + c_puct·√(log(parent_visits+1) / (visit_counts[i]+1))
2. selected = argmax(UCB)
3. rollout = torch.rand(1).item()  (random value in [0,1])
4. Q_new = (Q_old·n + rollout) / (n+1);  visit_counts[selected] += 1
```

## 参考答案

```python
def mcts_step(values, visit_counts, parent_visits, c_puct=1.414):
    # UCB1 scores
    exploration = c_puct * torch.sqrt(
        torch.log(torch.tensor(parent_visits + 1, dtype=torch.float32)) /
        (visit_counts.float() + 1)
    )
    ucb = values + exploration
    selected_idx = int(ucb.argmax().item())
    # Simulate rollout: random value in [0, 1]
    rollout_value = torch.rand(1).item()
    n = visit_counts[selected_idx].item()
    new_value = (values[selected_idx].item() * n + rollout_value) / (n + 1)
    updated_values = values.clone()
    updated_visits = visit_counts.clone()
    updated_values[selected_idx] = new_value
    updated_visits[selected_idx] += 1
    return selected_idx, updated_values, updated_visits
```

## 示例代码

```python
torch.manual_seed(42)
num_children = 5
values = torch.zeros(num_children)
visits = torch.zeros(num_children, dtype=torch.long)
parent_visits = 0

print("Running 5 MCTS steps:")
print(f"{'Step':>4}  {'Selected':>8}  {'Visit counts'}")
for step in range(5):
    idx, values, visits = mcts_step(values, visits, parent_visits)
    parent_visits += 1
    print(f"{step+1:>4}  {idx:>8}  {visits.tolist()}")

print(f"
Total visits: {visits.sum().item()} (expected 5)")
print(f"All nodes visited at least once: {(visits > 0).all().item()}")
```

