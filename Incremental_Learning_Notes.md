# 增量学习记录

> 复现过程中对比之前两轮新学习到的知识点
> 持续更新，记录深入理解与新发现

---

## 📋 目录

1. [基础组件深化](#1-基础组件深化)
2. [注意力机制深入](#2-注意力机制深入)
3. [训练技术实践](#3-训练技术实践)
4. [对齐技术理解](#4-对齐技术理解)
5. [推理优化细节](#5-推理优化细节)
6. [工程落地经验](#6-工程落地经验)
7. [踩坑与解决方案](#7-踩坑与解决方案)

---

## 1. 基础组件深化

### 1.1 Softmax 数值稳定性

**之前理解：** 知道要减去最大值，但不清楚具体原因

**深入学习：**
```
问题：exp(x) 在 x 很大时会溢出（FP16 最大值 65504）

示例：
x = [1000, 1001, 1002]
exp(x) = [inf, inf, inf]  # 溢出

解决方案：减去最大值
x_max = 1002
exp(x - x_max) = [exp(-2), exp(-1), exp(0)] = [0.135, 0.368, 1.0]  # 安全

数学等价性：
softmax(x) = exp(x) / sum(exp(x))
           = exp(x - c) * exp(c) / (sum(exp(x - c)) * exp(c))
           = exp(x - c) / sum(exp(x - c))
           = softmax(x - c)  # c 是任意常数
```

**新学到的点：**
- 减去最大值不仅防止溢出，还能提高精度
- log-sum-exp 技巧在很多地方都有应用（交叉熵、CRF 等）
- FP16 的数值范围比想象中更受限

---

### 1.2 RMSNorm vs LayerNorm

**之前理解：** RMSNorm 更简单，省去了均值减法

**深入学习：**
```
LayerNorm:
- 计算均值和方差
- 归一化：(x - mean) / sqrt(var + eps)
- 可学习参数：gamma, beta
- 计算量：2次 reduce 操作

RMSNorm:
- 只计算 RMS（均方根）
- 归一化：x / sqrt(mean(x^2) + eps)
- 可学习参数：只有 gamma
- 计算量：1次 reduce 操作

性能对比：
- RMSNorm 比 LayerNorm 快 10-15%
- 效果相当或更好
- 现代 LLM（LLaMA, Mistral）都用 RMSNorm
```

**新学到的点：**
- RMSNorm 的论文发现：LayerNorm 的成功主要来自缩放不变性，而非重新中心化
- 去掉均值减法可以减少计算量，且不影响效果
- 实际实现时，RMSNorm 可以融合到前一个算子中，进一步优化

---

### 1.3 SwiGLU 激活函数

**之前理解：** 知道是门控机制，但不清楚为什么比 ReLU 好

**深入学习：**
```
ReLU: f(x) = max(0, x)
- 问题：负半轴梯度为 0，神经元可能"死亡"

GELU: f(x) = x * Φ(x)
- 平滑近似 ReLU
- 负半轴有小的梯度

SwiGLU: f(x, W1, W2, Wgate) = (x @ W1) * swish(x @ Wgate)
- swish(z) = z * sigmoid(z)
- 门控机制：sigmoid(z) 控制信息流
- 效果：比 ReLU/GELU 提升 1-2%

为什么更好：
1. 门控机制提供了更灵活的信息流控制
2. swish 是平滑的，梯度更稳定
3. 实验表明，在相同参数量下效果更好
```

**新学到的点：**
- SwiGLU 的参数量比标准 MLP 多 50%（3 个投影 vs 2 个）
- 为了保持总参数量相同，d_ff 通常设为 2/3 * 4d
- LLaMA 的 d_ff = 11008（约 2/3 * 16384）

---

## 2. 注意力机制深入

### 2.1 Flash Attention 的在线 Softmax

**之前理解：** 知道分块计算，但不清楚在线更新的细节

**深入学习：**
```
标准 softmax 需要知道全局最大值：
softmax(x_i) = exp(x_i - max(x)) / Σexp(x_j - max(x))

问题：分块计算时，每块的最大值不同，如何合并？

在线更新算法：
初始化：m = -inf, l = 0, acc = 0

对于每个新块 scores_block:
1. block_max = max(scores_block)
2. new_max = max(m, block_max)
3. 修正之前累加器：
   acc = acc * exp(m - new_max)
   l = l * exp(m - new_max)
4. 处理新块：
   exp_scores = exp(scores_block - new_max)
   acc = acc + exp_scores @ V_block
   l = l + sum(exp_scores)
5. 更新最大值：
   m = new_max

最终输出：acc / l
```

**新学到的点：**
- 在线 softmax 是一个精妙的数学技巧
- 修正因子 `exp(m - new_max)` 保证了数值正确性
- 这个技巧也用在 Ring Attention、Sequence Parallelism 等场景

---

### 2.2 GQA 的 KV Cache 节省

**之前理解：** 知道 GQA 减少 KV 头数，但不清楚具体节省多少

**深入学习：**
```
MHA (Multi-Head Attention):
- Q heads: H
- KV heads: H
- KV Cache: 2 * H * d_k * seq_len

GQA (Grouped Query Attention):
- Q heads: H
- KV heads: G (G < H)
- KV Cache: 2 * G * d_k * seq_len
- 节省比例: 1 - G/H

MQA (Multi-Query Attention):
- Q heads: H
- KV heads: 1
- KV Cache: 2 * 1 * d_k * seq_len
- 节省比例: 1 - 1/H

示例（LLaMA 2 70B）：
- H = 64, G = 8 (GQA-8)
- KV Cache 节省: 1 - 8/64 = 87.5%
- 质量损失: < 0.5%
```

**新学到的点：**
- GQA 的关键洞察：不同 Q 头可以共享 KV 信息
- 从 MHA 转换到 GQA 可以通过 "uptraining" 实现
- 实际部署中，GQA 的 KV Cache 可以完全放入 L3 缓存

---

### 2.3 MLA 的低秩压缩

**之前理解：** 知道 MLA 压缩 KV，但不清楚压缩比和效果

**深入学习：**
```
MLA (Multi-Head Latent Attention):
- 压缩：c_kv = X @ W_dkv, 形状 (B, S, kv_rank)
- 解压：K = c_kv @ W_uk, V = c_kv @ W_uv

对比 GQA:
- GQA-8: KV Cache = 8 * d_k * seq_len
- MLA: KV Cache = kv_rank * seq_len
- 如果 kv_rank < 8 * d_k，MLA 更省

DeepSeek V2 的配置：
- d_model = 5120, num_heads = 128, d_k = 128
- GQA-8: KV Cache = 8 * 128 = 1024 维
- MLA: kv_rank = 512，节省 50%

额外优势：
- MLA 的表达能力更强（低秩解压可以学习更复杂的映射）
- 推理时只需缓存 c_kv，解压在计算时进行
```

**新学到的点：**
- MLA 是 DeepSeek V2/V3 的核心创新之一
- 低秩压缩不仅节省内存，还能提高表达能力
- 实现时需要特殊的 CUDA kernel 来优化解压过程

---

### 2.4 Flash Attention 的反向传播

**之前理解：** 知道要重计算，但不清楚具体实现

**深入学习：**
```
问题：前向传播时不存储注意力矩阵 S = Q @ K^T
      反向传播时需要 S 来计算梯度

解决方案：重计算（Recomputation）

前向传播：
- 只存储输出 O 和 softmax 的 logsumexp L
- 不存储中间结果 S, P

反向传播：
- 重新计算 S = Q @ K^T
- 重新计算 P = softmax(S)
- 计算梯度 dQ, dK, dV

内存节省：
- 标准注意力：存储 S (B, H, S, S) 和 P (B, H, S, S)
- Flash Attention：只存储 O (B, S, D) 和 L (B, S, 1)
- 节省：O(S²) -> O(S)

计算开销：
- 反向传播需要重新计算 S 和 P
- 额外计算量约 2x
- 但由于减少了内存访问，实际速度可能更快
```

**新学到的点：**
- 重计算是"以计算换内存"的经典策略
- 实际实现中，可以只重计算部分层，平衡内存和计算
- Flash Attention 2 进一步优化了重计算的效率

---

## 3. 训练技术实践

### 3.1 混合精度训练的 Loss Scaling

**之前理解：** 知道要放大 loss，但不清楚具体原因

**深入学习：**
```
问题：FP16 的梯度范围有限
- FP16 最小正规数：2^(-14) ≈ 6.1e-5
- 很多梯度值小于此，会下溢为 0

解决方案：Loss Scaling
1. 前向传播：正常计算
2. 计算 loss 后，乘以 scale_factor（如 1024）
3. 反向传播：梯度自动放大
4. 更新前：梯度除以 scale_factor，恢复正常

动态 Loss Scaling：
- 初始 scale_factor = 2^16
- 每 N 步尝试增大 scale_factor
- 如果出现 NaN/Inf，减小 scale_factor
- 使用 PyTorch 的 GradScaler 自动管理

示例：
原始 loss = 0.5
scaled loss = 0.5 * 1024 = 512
反向传播后，梯度放大 1024 倍
更新前，梯度缩小 1024 倍
```

**新学到的点：**
- Loss Scaling 是混合精度训练的关键技术
- 动态调整 scale_factor 可以适应不同的训练阶段
- BF16 不需要 Loss Scaling（因为范围与 FP32 相同）

---

### 3.2 FSDP 的分片策略

**之前理解：** 知道 FSDP 分片参数，但不清楚具体如何分片

**深入学习：**
```
FSDP (Fully Sharded Data Parallel) 的工作流程：

1. 初始化：
   - 将模型参数分成多个 flat parameter
   - 每个 worker 只保存一个 shard

2. 前向传播：
   - All-gather：收集完整参数
   - 计算完成后，释放非本 shard 的参数

3. 反向传播：
   - All-gather：再次收集完整参数
   - 计算梯度
   - Reduce-scatter：聚合梯度，每个 worker 保留自己的 shard

4. 参数更新：
   - 每个 worker 只更新自己的 shard

内存节省（假设 world_size = N）：
- 参数：每个 worker 1/N
- 梯度：每个 worker 1/N
- 优化器状态：每个 worker 1/N
- 总节省：约 N 倍
```

**新学到的点：**
- FSDP 的通信量与 DDP 相同（1.5x）
- 但内存节省是线性的
- 需要权衡通信开销和内存节省

---

### 3.3 激活检查点的选择

**之前理解：** 知道可以节省内存，但不清楚应该在哪里使用

**深入学习：**
```
激活检查点的权衡：
- 内存节省：O(L) -> O(√L)（L 是层数）
- 计算开销：约 33%（需要重新前向传播）

选择策略：
1. 检查点放在 Transformer Block 边界
2. 不要在每个小操作后都检查点
3. 计算密集的层（如 Attention）不要检查点

实际配置（LLaMA 训练）：
- 每个 Transformer Block 作为一个检查点
- 32 层模型，检查点间隔 1 层
- 内存节省：约 50%
- 计算开销：约 20%
```

**新学到的点：**
- 检查点的位置选择很重要
- 太频繁会增加计算开销，太稀疏会增加内存
- 实际使用时，通常在 Transformer Block 级别设置

---

## 4. 对齐技术理解

### 4.1 DPO 的数学推导

**之前理解：** 知道 DPO 的损失函数，但不清楚如何推导

**深入学习：**
```
从 RLHF 到 DPO 的推导：

RLHF 目标：
max E[reward(x,y)] - β * KL(π || π_ref)

最优策略的闭式解：
π*(y|x) = π_ref(y|x) * exp(reward(x,y)/β) / Z(x)

其中 Z(x) = Σ_y π_ref(y|x) * exp(reward(x,y)/β)

反解 reward：
reward(x,y) = β * log(π*(y|x) / π_ref(y|x)) + β * log(Z(x))

代入 Bradley-Terry 模型：
P(y_w > y_l | x) = σ(reward(x,y_w) - reward(x,y_l))

关键：Z(x) 在两个 reward 的差中消除了！

P(y_w > y_l | x) = σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))

取负对数，得到 DPO 损失：
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

**新学到的点：**
- DPO 的核心洞察：reward 可以用策略模型和参考模型的对数概率比表示
- Z(x) 的消除是推导的关键
- DPO 的梯度会同时增加 chosen 的概率、降低 rejected 的概率

---

### 4.2 GRPO 的组内归一化

**之前理解：** 知道 GRPO 用组内归一化，但不清楚为什么

**深入学习：**
```
问题：奖励的绝对值没有意义，相对值才有意义

示例：
- 两个回复的奖励：0.8 和 0.6
- 差值 0.2 表示什么？
- 如果其他回复的奖励都是 0.1-0.3，那 0.8 和 0.6 都很好
- 如果其他回复的奖励都是 0.9-1.0，那 0.8 和 0.6 都一般

解决方案：组内归一化
- 对同一个 prompt 的多个回复，计算组内均值和标准差
- 归一化：advantage = (reward - mean) / std
- 这样得到的是相对优势，而不是绝对奖励

PPO vs GRPO：
- PPO：需要训练 Critic 网络估计 value function
- GRPO：用组内统计量代替 Critic，更简单
- 效果：GRPO 在 DeepSeek 中表现与 PPO 相当
```

**新学到的点：**
- GRPO 的核心洞察：奖励是相对的，不是绝对的
- 组内归一化天然提供了 baseline，无需额外的 Critic
- 这个技巧在 bandit 问题中很常见

---

### 4.3 PPO 的裁剪机制

**之前理解：** 知道 PPO 裁剪比率，但不清楚为什么有效

**深入学习：**
```
问题：策略梯度的高方差

重要性采样：
- 用旧策略的数据估计新策略的梯度
- 比率：r = π_new(a|s) / π_old(a|s)
- 如果 r 很大或很小，方差会很高

PPO 的解决方案：裁剪
- 目标：min(r * A, clip(r, 1-ε, 1+ε) * A)
- 如果 A > 0（好动作）：r 被限制在 [1-ε, 1+ε]
- 如果 A < 0（坏动作）：r 被限制在 [1-ε, 1+ε]

效果：
- 防止策略更新过大
- 降低方差，提高稳定性
- ε 通常取 0.1-0.2
```

**新学到的点：**
- 裁剪是一种信任域约束
- 不是简单的梯度裁剪，而是目标函数的裁剪
- 实现时需要同时计算裁剪和未裁剪的目标，取较小值

---

## 5. 推理优化细节

### 5.1 KV Cache 的内存管理

**之前理解：** 知道 KV Cache 缓存之前的计算，但不清楚管理细节

**深入学习：**
```
KV Cache 的内存占用：
- 每层：2 * num_heads * d_head * seq_len * dtype_size
- 总共：2 * num_layers * num_heads * d_head * seq_len * dtype_size

示例（LLaMA 7B）：
- num_layers = 32, num_heads = 32, d_head = 128
- FP16：2 bytes
- seq_len = 2048
- 每个请求：2 * 32 * 32 * 128 * 2048 * 2 = 1GB

问题：
- 不同请求的序列长度不同
- 预分配最大长度会浪费内存
- 动态分配会导致内存碎片

解决方案：Paged Attention
- 将 KV Cache 分成固定大小的页（如 16 tokens）
- 使用 block table 映射逻辑位置到物理位置
- 按需分配，减少碎片
```

**新学到的点：**
- KV Cache 是推理的主要内存瓶颈
- Paged Attention 借鉴了操作系统的虚拟内存思想
- 实际实现需要考虑页的分配、回收、碎片整理

---

### 5.2 推测解码的接受率

**之前理解：** 知道推测解码用小模型提议，但不清楚接受率如何计算

**深入学习：**
```
推测解码的接受概率：
- 草稿模型提议 token t
- 目标模型的概率：p_target(t)
- 草稿模型的概率：p_draft(t)
- 接受概率：min(1, p_target(t) / p_draft(t))

直觉：
- 如果目标模型更倾向接受这个 token，接受概率高
- 如果草稿模型过于自信，接受概率低

拒绝时的重采样：
- 从 max(0, p_target - p_draft) 归一化后采样
- 保证最终分布与目标模型一致

期望接受长度：
- 如果草稿模型与目标模型相似，接受率高
- 如果差异大，接受率低
- 平均接受长度 ≈ 1 / (1 - α)，其中 α 是平均接受率
```

**新学到的点：**
- 推测解码的核心是接受-拒绝采样
- 数学上可以证明，最终分布与目标模型完全一致
- 草稿模型的选择很重要：太差接受率低，太好没必要

---

### 5.3 INT8 量化的精度损失

**之前理解：** 知道量化会损失精度，但不清楚具体影响

**深入学习：**
```
INT8 量化公式：
- scale = max(abs(weight)) / 127
- quantized = round(weight / scale)
- dequantized = quantized * scale

精度损失来源：
1. 舍入误差：round 操作引入的误差
2. 截断误差：超出 [-128, 127] 范围的值被截断
3. 分布不匹配：如果权重分布不均匀，量化效果差

实际影响：
- LLaMA 7B：INT8 量化后，困惑度增加约 0.1-0.2
- 生成质量：几乎无感知差异
- 速度提升：约 1.5-2x

进一步量化：INT4 / GPTQ
- 更激进的量化，精度损失更大
- 需要更复杂的量化策略（如分组量化）
- 速度提升：约 3-4x
```

**新学到的点：**
- 量化的核心是找到最优的 scale factor
- 不同层对量化的敏感度不同（第一层和最后一层更敏感）
- 混合精度量化（某些层保持高精度）可以平衡速度和精度

---

## 6. 工程落地经验

### 6.1 分布式训练的通信开销

**之前理解：** 知道分布式训练有通信开销，但不清楚具体多少

**深入学习：**
```
DDP (Distributed Data Parallel) 的通信：
- 每次反向传播后，All-reduce 梯度
- 通信量：2 * 参数量 * dtype_size（一次 reduce，一次 broadcast）
- 通信时间：取决于网络带宽和延迟

FSDP 的通信：
- 前向传播：All-gather 参数
- 反向传播：All-gather 参数 + Reduce-scatter 梯度
- 通信量：约 1.5x DDP

优化策略：
1. 梯度累积：减少通信频率
2. 通信计算重叠：在计算的同时进行通信
3. 压缩通信：使用 FP16 梯度
4. 层次化通信：先节点内，再节点间

实际测量（8 卡 3090，PCIe）：
- 模型大小 1B，FP16
- All-reduce 时间：约 50ms
- 占总训练时间的 10-20%
```

**新学到的点：**
- 通信开销是分布式训练的主要瓶颈
- NVLink vs PCIe 的带宽差异很大（600GB/s vs 32GB/s）
- 梯度累积是最简单有效的优化

---

### 6.2 混合精度训练的实践

**之前理解：** 知道要用 FP16，但不清楚具体配置

**深入学习：**
```
PyTorch 混合精度训练的标准流程：

scaler = torch.cuda.amp.GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    
    # 前向传播（FP16）
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 反向传播（FP16 梯度）
    scaler.scale(loss).backward()
    
    # 更新参数（FP32）
    scaler.step(optimizer)
    scaler.update()

关键点：
1. autocast 自动将适合的操作转为 FP16
2. GradScaler 自动管理 loss scaling
3. 优化器更新 FP32 主权重

不适合 FP16 的操作：
- Softmax（需要大范围）
- LayerNorm（需要高精度）
- 损失计算（需要高精度）
```

**新学到的点：**
- autocast 会自动判断哪些操作适合 FP16
- GradScaler 会自动调整 scale factor
- BF16 不需要 GradScaler，因为范围与 FP32 相同

---

## 7. 踩坑与解决方案

### 7.1 梯度爆炸问题

**问题描述：**
- 训练过程中 loss 突然变成 NaN
- 梯度范数突然变大

**排查过程：**
```python
# 1. 监控梯度范数
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"Large gradient in {name}: {grad_norm}")

# 2. 找到问题层
# 通常是 attention 的输出投影或 MLP 的最后一层

# 3. 检查初始化
# 确认使用了正确的初始化方法
```

**解决方案：**
1. 梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
2. 学习率预热：前 1000 步线性增加学习率
3. 权重衰减：添加 L2 正则化
4. 检查初始化：使用 Xavier/Kaiming 初始化

---

### 7.2 内存泄漏问题

**问题描述：**
- 训练过程中显存持续增加
- 最终 OOM

**排查过程：**
```python
# 1. 监控显存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 2. 检查是否有张量未释放
import gc
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(f"Tensor: {obj.shape}, {obj.device}")
```

**常见原因：**
1. loss.item() 未调用，保留了计算图
2. 日志记录了张量，导致引用未释放
3. 缓存未清理

**解决方案：**
```python
# 正确的 loss 记录
loss_val = loss.item()  # 释放计算图
writer.add_scalar('loss', loss_val, step)

# 定期清理缓存
torch.cuda.empty_cache()
```

---

### 7.3 多卡训练不同步

**问题描述：**
- 不同卡的模型参数不一致
- 训练结果不可复现

**排查过程：**
```python
# 1. 检查随机种子
print(f"Seed: {torch.initial_seed()}")
print(f"CUDA seed: {torch.cuda.initial_seed()}")

# 2. 检查数据加载
# 确认每个 worker 使用不同的数据

# 3. 检查模型同步
# 打印不同卡的参数，确认一致
```

**解决方案：**
```python
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 使用 DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 确保模型同步
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## 📊 学习进度跟踪

| 阶段 | 状态 | 新知识点数量 | 关键收获 |
|------|------|-------------|---------|
| 基础组件 | 进行中 | 8 | 数值稳定性、RMSNorm、SwiGLU |
| 注意力机制 | 进行中 | 12 | Flash Attention、GQA、MLA |
| 训练技术 | 进行中 | 10 | 混合精度、FSDP、激活检查点 |
| 对齐技术 | 进行中 | 8 | DPO 推导、GRPO、PPO 裁剪 |
| 推理优化 | 进行中 | 6 | KV Cache、推测解码、量化 |
| 工程落地 | 进行中 | 5 | 通信优化、内存管理 |

---

**最后更新：2026 年 5 月 27 日**
**基于 8 × RTX 3090 复现过程中的增量学习记录**