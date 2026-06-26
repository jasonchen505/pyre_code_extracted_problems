# Pyre Code 项目复现计划

> 基于 8 卡 RTX 3090 (24GB × 8 = 192GB) 的完整复现方案
> 目标：系统性掌握 LLM & Agent & RL & 后训练核心技术

---

## 📋 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [复现策略总览](#2-复现策略总览)
3. [分阶段执行计划](#3-分阶段执行计划)
4. [详细任务清单](#4-详细任务清单)
5. [验证与测试方案](#5-验证与测试方案)
6. [时间规划](#6-时间规划)

---

## 1. 硬件资源评估

### 1.1 硬件规格

| 资源 | 规格 | 说明 |
|------|------|------|
| GPU | 8 × RTX 3090 | 每卡 24GB 显存 |
| 总显存 | 192GB | 可支持 7B 模型训练 |
| FP32 算力 | 35.6 TFLOPS/卡 | 总计 284.8 TFLOPS |
| FP16 算力 | 71 TFLOPS/卡 | 总计 568 TFLOPS |
| NVLink | 无（假设） | 卡间通信走 PCIe |

### 1.2 资源约束分析

**可以做的事情：**
- ✅ 所有基础组件实现与验证（无需 GPU 或单卡）
- ✅ 小模型训练（125M-1B 参数）
- ✅ 中等模型训练（1B-7B 参数，需要 FSDP/混合精度）
- ✅ LoRA/QLoRA 微调（7B 模型）
- ✅ DPO/GRPO 对齐训练（7B 模型）
- ✅ 分布式训练技术验证（FSDP、张量并行）
- ✅ 推理优化验证（KV Cache、量化、推测解码）

**受限但可行：**
- ⚠️ 7B 全量微调（需要 FSDP + 激活检查点 + 混合精度）
- ⚠️ 13B 模型训练（显存紧张，需要激进优化）
- ⚠️ 大规模 RLHF（奖励模型 + 策略模型同时加载）

**不可行：**
- ❌ 70B+ 模型训练（显存不足）
- ❌ 大规模 MoE 训练（专家数量受限）
- ❌ 长序列训练（>32K，显存瓶颈）

### 1.3 显存预算估算

**7B 模型的显存需求：**
```
模型参数：7B × 2 bytes (FP16) = 14GB
梯度：7B × 2 bytes = 14GB
优化器状态（Adam）：7B × 8 bytes = 56GB
激活值：取决于 batch size 和序列长度
--------------------------------------
总计（不含激活）：~84GB
单卡 24GB 不够，需要至少 4 卡 FSDP
```

**使用 FSDP + 混合精度 + 激活检查点：**
```
FSDP 分片后每卡：84GB / 4 = 21GB（4 卡）
加上激活检查点：每卡额外 2-4GB
总计：~25GB/卡（4 卡配置可行）
```

---

## 2. 复现策略总览

### 2.1 核心原则

1. **先简后繁**：先实现基础组件，再组合成复杂系统
2. **先验证后扩展**：先在小规模验证正确性，再扩展到大规模
3. **理论结合实践**：每个组件都要理解原理 + 动手实现 + 验证效果
4. **记录增量知识**：每阶段记录新学到的点

### 2.2 复现范围

| 类别 | 题目数量 | GPU 需求 | 优先级 |
|------|---------|---------|--------|
| 基础组件 | 15 | 无/单卡 | P0 |
| 注意力机制 | 10 | 单卡 | P0 |
| 位置编码 | 4 | 单卡 | P0 |
| 训练技术 | 8 | 多卡 | P0 |
| 对齐技术 | 8 | 多卡 | P0 |
| 推理优化 | 7 | 单卡 | P1 |
| 参数高效微调 | 2 | 多卡 | P0 |
| 分布式训练 | 3 | 多卡 | P0 |
| 架构组件 | 6 | 单卡 | P1 |
| 扩散模型 | 4 | 单卡 | P2 |
| Agent 推理 | 2 | 单卡 | P1 |
| 图神经网络 | 8 | 单卡 | P2 |

### 2.3 复现路径

```
阶段 1：基础组件（1-2 周）
    ↓
阶段 2：注意力与位置编码（1-2 周）
    ↓
阶段 3：完整 Transformer 训练（2-3 周）
    ↓
阶段 4：分布式训练与优化（2-3 周）
    ↓
阶段 5：对齐技术（2-3 周）
    ↓
阶段 6：推理优化与部署（1-2 周）
    ↓
阶段 7：综合项目（2-4 周）
```

---

## 3. 分阶段执行计划

### 阶段 1：基础组件实现（Week 1-2）

**目标：** 掌握 LLM 的基础构建模块

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 1.1 | ReLU / GELU / SwiGLU 激活函数 | Easy | 无 | 2h |
| 1.2 | Softmax（数值稳定版本） | Easy | 无 | 2h |
| 1.3 | LayerNorm / RMSNorm | Medium | 无 | 3h |
| 1.4 | Embedding 层 | Easy | 无 | 1h |
| 1.5 | Linear 层（含 Kaiming 初始化） | Medium | 无 | 2h |
| 1.6 | Dropout | Easy | 无 | 1h |
| 1.7 | Cross Entropy Loss | Easy | 无 | 2h |
| 1.8 | Label Smoothing | Easy | 无 | 2h |
| 1.9 | Focal Loss | Medium | 无 | 2h |
| 1.10 | Adam 优化器 | Medium | 无 | 3h |
| 1.11 | Cosine LR Scheduler | Medium | 无 | 2h |
| 1.12 | Gradient Clipping | Easy | 无 | 1h |
| 1.13 | Gradient Accumulation | Easy | 无 | 1h |
| 1.14 | Mixed Precision Training | Medium | 单卡 | 3h |
| 1.15 | Activation Checkpointing | Medium | 单卡 | 2h |

**验证标准：**
- 每个组件的输出与 PyTorch 官方实现一致（误差 < 1e-5）
- 梯度计算正确
- 性能测试通过

**学习重点：**
- 数值稳定性处理（softmax、cross entropy）
- 内存优化技术（激活检查点、混合精度）
- 优化器原理（Adam 的偏差校正）

---

### 阶段 2：注意力与位置编码（Week 3-4）

**目标：** 掌握 Transformer 的核心计算

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 2.1 | Scaled Dot-Product Attention | Easy | 单卡 | 2h |
| 2.2 | Multi-Head Attention | Medium | 单卡 | 3h |
| 2.3 | Causal Attention | Medium | 单卡 | 2h |
| 2.4 | Cross Attention | Medium | 单卡 | 2h |
| 2.5 | Grouped Query Attention (GQA) | Hard | 单卡 | 4h |
| 2.6 | Multi-Head Latent Attention (MLA) | Hard | 单卡 | 4h |
| 2.7 | Differential Attention | Hard | 单卡 | 3h |
| 2.8 | Sliding Window Attention | Medium | 单卡 | 2h |
| 2.9 | Linear Attention | Hard | 单卡 | 3h |
| 2.10 | Flash Attention | Hard | 单卡 | 6h |
| 2.11 | Sinusoidal Position Encoding | Easy | 无 | 1h |
| 2.12 | RoPE | Medium | 单卡 | 3h |
| 2.13 | ALiBi | Medium | 单卡 | 2h |
| 2.14 | NTK-aware RoPE | Medium | 单卡 | 2h |

**验证标准：**
- 注意力输出与标准实现一致
- 因果掩码正确应用
- 位置编码的数学性质验证（相对位置、范数保持）

**学习重点：**
- 在线 softmax 的原理与实现
- KV Cache 优化的动机
- 不同位置编码的优劣对比

---

### 阶段 3：完整 Transformer 训练（Week 5-7）

**目标：** 从零实现并训练一个小型 LLM

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 3.1 | GPT-2 Block 实现 | Hard | 单卡 | 4h |
| 3.2 | SwiGLU MLP 实现 | Medium | 单卡 | 2h |
| 3.3 | 完整 GPT 模型组装 | Hard | 单卡 | 6h |
| 3.4 | 数据预处理（BPE 分词） | Hard | 无 | 4h |
| 3.5 | 训练循环实现 | Medium | 多卡 | 4h |
| 3.6 | 训练 125M 模型 | Medium | 4 卡 | 24h |
| 3.7 | 评估与分析 | Medium | 单卡 | 4h |

**模型配置（125M）：**
```python
config = {
    "vocab_size": 50257,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "batch_size": 16,  # 每卡
    "learning_rate": 6e-4,
    "max_iters": 100000,
}
```

**显存估算：**
- 模型参数：125M × 2 bytes = 250MB
- 优化器状态：125M × 8 bytes = 1GB
- 激活值：~2GB（batch_size=16, seq_len=1024）
- 总计：~4GB/卡（单卡即可）

**学习重点：**
- Pre-norm vs Post-norm 的区别
- 训练稳定性技巧（warmup、梯度裁剪）
- 损失曲线分析

---

### 阶段 4：分布式训练与优化（Week 8-10）

**目标：** 掌握大规模模型训练技术

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 4.1 | 张量并行 MLP | Hard | 多卡 | 6h |
| 4.2 | FSDP 训练步骤 | Hard | 多卡 | 6h |
| 4.3 | Ring Attention | Hard | 多卡 | 6h |
| 4.4 | 数据并行训练 | Medium | 多卡 | 4h |
| 4.5 | 混合精度 + FSDP 联合训练 | Hard | 多卡 | 8h |
| 4.6 | 训练 1B 模型 | Hard | 8 卡 | 48h |
| 4.7 | 性能分析与优化 | Medium | 多卡 | 8h |

**1B 模型配置：**
```python
config = {
    "vocab_size": 50257,
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 2048,
    "block_size": 2048,
    "batch_size": 4,  # 每卡，使用梯度累积
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-4,
    "max_iters": 50000,
}
```

**显存估算（FSDP 4 卡）：**
- 模型参数：1B × 2 bytes = 2GB
- 优化器状态：1B × 8 bytes = 8GB
- FSDP 分片后每卡：(2+8) / 4 = 2.5GB
- 激活检查点后激活值：~4GB/卡
- 总计：~7GB/卡（4 卡配置）

**学习重点：**
- FSDP 的 All-gather 和 Reduce-scatter
- 张量并行的通信模式
- 混合精度训练的 Loss Scaling

---

### 阶段 5：对齐技术（Week 11-13）

**目标：** 掌握 LLM 对齐的主流方法

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 5.1 | Reward Model 训练 | Medium | 多卡 | 8h |
| 5.2 | DPO 损失实现 | Medium | 单卡 | 4h |
| 5.3 | GRPO 损失实现 | Hard | 单卡 | 4h |
| 5.4 | PPO 损失实现 | Hard | 单卡 | 4h |
| 5.5 | Contrastive Loss | Medium | 单卡 | 2h |
| 5.6 | DPO 训练 1B 模型 | Hard | 8 卡 | 24h |
| 5.7 | GRPO 训练 1B 模型 | Hard | 8 卡 | 24h |
| 5.8 | 评估与对比分析 | Medium | 单卡 | 8h |

**DPO 训练配置：**
```python
config = {
    "base_model": "path/to/sft-1b",
    "beta": 0.1,
    "batch_size": 2,  # 每卡
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-7,
    "max_steps": 10000,
}
```

**显存估算：**
- 策略模型：1B × 2 bytes = 2GB
- 参考模型：1B × 2 bytes = 2GB
- 优化器状态：~2GB
- FSDP 分片后每卡：~3GB
- 总计：~8GB/卡（8 卡配置）

**学习重点：**
- DPO 的数学推导
- GRPO 的组内归一化
- PPO 的裁剪机制
- 偏好数据的收集与处理

---

### 阶段 6：推理优化与部署（Week 14-15）

**目标：** 掌握 LLM 推理优化技术

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 6.1 | KV Cache 实现 | Hard | 单卡 | 6h |
| 6.2 | Top-k / Top-p 采样 | Medium | 单卡 | 2h |
| 6.3 | Beam Search | Medium | 单卡 | 3h |
| 6.4 | 推测解码 | Hard | 单卡 | 6h |
| 6.5 | INT8 量化 | Hard | 单卡 | 4h |
| 6.6 | Paged Attention | Hard | 单卡 | 6h |
| 6.7 | LoRA 实现 | Medium | 单卡 | 4h |
| 6.8 | QLoRA 实现 | Hard | 单卡 | 4h |
| 6.9 | 推理服务搭建 | Medium | 多卡 | 8h |

**学习重点：**
- KV Cache 的内存管理
- 量化的精度-速度权衡
- LoRA 的低秩假设

---

### 阶段 7：综合项目（Week 16-19）

**目标：** 整合所有技术，完成端到端项目

**任务清单：**

| 序号 | 任务 | 难度 | GPU 需求 | 预计时间 |
|------|------|------|---------|---------|
| 7.1 | MoE 架构实现 | Hard | 多卡 | 8h |
| 7.2 | Mamba SSM 实现 | Hard | 单卡 | 6h |
| 7.3 | MCTS 推理 | Hard | 单卡 | 6h |
| 7.4 | Multi-Token Prediction | Hard | 单卡 | 4h |
| 7.5 | 端到端训练流程 | Hard | 8 卡 | 48h |
| 7.6 | 推理服务部署 | Hard | 多卡 | 16h |
| 7.7 | 性能评估与优化 | Medium | 多卡 | 16h |

---

## 4. 详细任务清单

### 4.1 基础组件任务

#### 任务 1.1：ReLU / GELU / SwiGLU 激活函数

**实现要点：**
```python
# ReLU
def relu(x):
    return x * (x > 0).float()

# GELU
def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

# SwiGLU
def swiglu(x, W1, W2, Wgate):
    gate = x @ Wgate
    swish_gate = gate * torch.sigmoid(gate)
    return (x @ W1 * swish_gate) @ W2
```

**验证方法：**
```python
# 与 PyTorch 官方实现对比
assert torch.allclose(relu(x), F.relu(x), atol=1e-6)
assert torch.allclose(gelu(x), F.gelu(x), atol=1e-4)
```

**学习要点：**
- ReLU 的梯度稀疏性
- GELU 的平滑性优势
- SwiGLU 的门控机制

---

#### 任务 1.2：Softmax（数值稳定版本）

**实现要点：**
```python
def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

**为什么需要减去最大值：**
- 防止 exp 溢出
- 数学上等价，但数值更稳定

**学习要点：**
- log-sum-exp 技巧
- 数值稳定性的工程实践

---

#### 任务 1.10：Adam 优化器

**实现要点：**
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
```

**学习要点：**
- 偏差校正的必要性
- 一阶矩（动量）和二阶矩（自适应学习率）的作用
- AdamW 的权重衰减

---

### 4.2 注意力机制任务

#### 任务 2.5：Grouped Query Attention (GQA)

**实现要点：**
```python
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 扩展 KV 头以匹配 Q 头
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

**学习要点：**
- KV Cache 大小的计算
- MHA vs MQA vs GQA 的权衡
- repeat_interleave 的实现细节

---

#### 任务 2.10：Flash Attention

**实现要点：**
```python
def flash_attention(Q, K, V, block_size=32):
    B, S, D = Q.shape
    output = torch.zeros_like(Q)
    for i in range(0, S, block_size):
        qi = Q[:, i:i+block_size]
        bs_q = qi.shape[1]
        row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
        row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
        acc = torch.zeros(B, bs_q, D, device=Q.device)
        for j in range(0, S, block_size):
            kj = K[:, j:j+block_size]
            vj = V[:, j:j+block_size]
            scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            acc = acc * correction + torch.bmm(exp_scores, vj)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max
        output[:, i:i+block_size] = acc / row_sum
    return output
```

**学习要点：**
- 在线 softmax 的原理
- 分块计算的内存优势
- 重计算 vs 存储的权衡

---

### 4.3 对齐技术任务

#### 任务 5.2：DPO 损失实现

**实现要点：**
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    diff = chosen_rewards - rejected_rewards
    return -torch.log(torch.sigmoid(diff)).mean()
```

**学习要点：**
- DPO 的数学推导
- 参考模型的作用
- beta 参数的影响

---

#### 任务 5.3：GRPO 损失实现

**实现要点：**
```python
def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)
    
    advantages_detached = advantages.detach()
    return -(advantages_detached * logps).mean()
```

**学习要点：**
- 组内归一化的意义
- 与 PPO 的对比
- DeepSeek 的设计选择

---

## 5. 验证与测试方案

### 5.1 单元测试

每个组件都需要通过以下测试：
1. **数值正确性**：与 PyTorch 官方实现对比
2. **梯度正确性**：验证梯度计算
3. **形状正确性**：输出形状符合预期
4. **边界情况**：空输入、极大值、极小值

### 5.2 集成测试

1. **组件组合**：多个组件组合后的正确性
2. **端到端训练**：完整训练流程的正确性
3. **分布式训练**：多卡训练的一致性

### 5.3 性能测试

1. **内存占用**：监控 GPU 显存使用
2. **计算速度**：对比不同实现的吞吐量
3. **扩展性**：多卡的加速比

---

## 6. 时间规划

### 6.1 总体时间线

| 阶段 | 时间 | 主要任务 | 里程碑 |
|------|------|---------|--------|
| 阶段 1 | Week 1-2 | 基础组件 | 所有基础组件通过测试 |
| 阶段 2 | Week 3-4 | 注意力与位置编码 | 实现 Flash Attention |
| 阶段 3 | Week 5-7 | 完整 Transformer | 训练 125M 模型 |
| 阶段 4 | Week 8-10 | 分布式训练 | 训练 1B 模型 |
| 阶段 5 | Week 11-13 | 对齐技术 | 完成 DPO/GRPO 训练 |
| 阶段 6 | Week 14-15 | 推理优化 | 搭建推理服务 |
| 阶段 7 | Week 16-19 | 综合项目 | 端到端系统 |

### 6.2 每周安排

**工作日（周一至周五）：**
- 上午：学习理论，阅读论文
- 下午：动手实现，调试代码
- 晚上：总结记录，更新文档

**周末：**
- 复习本周内容
- 完成未完成的任务
- 记录增量知识

### 6.3 里程碑检查点

| 检查点 | 时间 | 验收标准 |
|--------|------|---------|
| M1 | Week 2 | 15 个基础组件全部通过测试 |
| M2 | Week 4 | Flash Attention 实现正确且有加速 |
| M3 | Week 7 | 125M 模型训练收敛 |
| M4 | Week 10 | 1B 模型 FSDP 训练成功 |
| M5 | Week 13 | DPO/GRPO 对齐训练完成 |
| M6 | Week 15 | 推理服务可运行 |
| M7 | Week 19 | 端到端系统完成 |

---

## 📊 资源使用建议

### 训练任务的 GPU 分配

| 任务 | GPU 数量 | 配置说明 |
|------|---------|---------|
| 125M 训练 | 1 单卡 | 纯数据并行即可 |
| 1B 训练 | 4 卡 | FSDP + 激活检查点 |
| 7B LoRA | 4 单卡 | INT8 量化 + LoRA |
| 1B DPO | 8 卡 | FSDP + 混合精度 |
| 推理服务 | 2-4 卡 | 模型并行 + KV Cache |

### 显存优化技巧

1. **混合精度训练**：使用 bf16，节省 50% 显存
2. **激活检查点**：以计算换内存，节省 60-80% 激活显存
3. **梯度累积**：减小每卡 batch size
4. **FSDP**：将模型参数分片到多卡
5. **INT8/INT4 量化**：推理时使用量化模型

---

**最后更新：2026 年 5 月 27 日**
**基于 8 × RTX 3090 (24GB × 8) 硬件配置**