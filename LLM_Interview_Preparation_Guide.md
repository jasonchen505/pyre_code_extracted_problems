# LLM 算法实习面试准备指南

> 基于 Pyre Code 项目 76 道 ML 编程题目的深度面试准备文档
> 适用于 LLM & Agent 应用、后训练方向的算法实习面试

---

## 📋 目录

1. [项目概述与学习路径](#1-项目概述与学习路径)
2. [核心题目分类与面试要点](#2-核心题目分类与面试要点)
3. [深度挖掘问题与考察点](#3-深度挖掘问题与考察点)
4. [综合应用场景与系统设计](#4-综合应用场景与系统设计)
5. [面试技巧与准备建议](#5-面试技巧与准备建议)

---

## 1. 项目概述与学习路径

### 1.1 项目简介

Pyre Code 是一个 ML 编程练习项目，包含 76 道题目，覆盖了现代 AI 系统的核心组件。对于 LLM 算法实习面试，这些题目提供了从基础到高级的完整知识体系。

### 1.2 推荐学习路径

```
基础层：激活函数 → 归一化 → 注意力机制 → 位置编码
    ↓
架构层：Transformer Block → MoE → Mamba SSM
    ↓
训练层：优化器 → 学习率调度 → 分布式训练 → 混合精度
    ↓
对齐层：奖励模型 → DPO/GRPO/PPO → RLHF
    ↓
推理层：KV Cache → 量化 → 投机解码 → 分页注意力
    ↓
应用层：LoRA/QLoRA → Agent 推理 → 多 Token 预测
```

### 1.3 面试重点标记

- 🔥 **高频考点**：几乎必问
- ⭐ **重要考点**：经常考察
- 📚 **基础知识**：需要掌握但较少深挖
- 💡 **加分项**：能答出来是加分

---

## 2. 核心题目分类与面试要点

### 2.1 注意力机制（Attention Mechanisms）

#### 🔥 Flash Attention（flash_attention.md）
**面试要点：**
- 在线 softmax（Online Softmax）的核心思想：如何在不存储完整注意力矩阵的情况下计算 softmax
- 分块处理（Tiling）的实现细节：如何处理非对齐序列长度
- 内存复杂度从 O(S²) 降至 O(S) 的原理
- 与标准注意力的数值等价性保证

**深挖问题：**
1. 为什么需要在线 softmax？直接分块计算会有什么问题？
2. 如何处理 block_size 不能整除序列长度的情况？
3. Flash Attention 的反向传播是如何实现的？（recomputation vs stored）
4. Flash Attention 2 相比 Flash Attention 1 有哪些改进？

#### 🔥 Grouped Query Attention (GQA)（gqa.md）
**面试要点：**
- KV Cache 大小与推理速度的权衡
- `repeat_interleave` 扩展 KV 头的实现
- 与 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 的关系
- LLaMA 2/3 中 GQA 的应用

**深挖问题：**
1. GQA 中 `num_kv_heads=1` 时退化为什么？（MQA）
2. 为什么 GQA 能在保持质量的同时减少 KV Cache？
3. 如何从预训练的 MHA 模型转换到 GQA？（Uptraining）
4. GQA 的 KV Cache 大小是多少？如何计算？

#### ⭐ Multi-Head Latent Attention (MLA)（mla.md）
**面试要点：**
- DeepSeek V2/V3 的核心创新
- 低秩压缩 KV Cache 的原理：`c_kv = X @ W_dkv`
- 推理时 KV Cache 只需存储 `c_kv`，大幅减少内存
- 与 GQA 的对比优势

**深挖问题：**
1. MLA 的 KV Cache 大小是多少？与 GQA 对比如何？
2. 为什么低秩压缩能保持注意力质量？
3. MLA 的训练成本与 GQA 相比如何？
4. DeepSeek V2 为什么选择 MLA 而不是 GQA？

#### ⭐ Differential Attention（diff_attention.md）
**面试要点：**
- ICLR 2025 论文《Differential Transformer》的核心思想
- 两个 softmax 注意力图相减消除噪声
- 可学习的 lambda 参数控制噪声消除强度
- 对长上下文建模的改进

**深挖问题：**
1. 为什么两个注意力图相减能消除噪声？
2. lambda 参数应该如何初始化？
3. 与标准注意力相比，计算量增加了多少？
4. Differential Attention 对哪些任务效果提升最明显？

#### ⭐ Sliding Window Attention（sliding_window.md）
**面试要点：**
- 局部注意力的实现：`|i-j| > window_size` 的位置用 `-inf` 掩盖
- Mistral 等模型中的应用
- 与全注意力的复杂度对比
- 滑动窗口大小的选择对性能的影响

**深挖问题：**
1. 滑动窗口注意力如何处理因果掩码？
2. 多层滑动窗口注意力的感受野是如何增长的？
3. 为什么 Mistral 选择滑动窗口而不是全注意力？
4. 滑动窗口注意力的 KV Cache 管理有什么特殊之处？

#### 📚 线性注意力（linear_attention.md）
**面试要点：**
- 核特征映射替代 softmax：`phi(x) = elu(x) + 1`
- 复杂度从 O(S²D) 降至 O(SD²) 的原理
- 重排计算顺序的技巧：`phi(Q) @ (phi(K)^T @ V)`

**深挖问题：**
1. 为什么线性注意力没有成为主流？
2. 线性注意力的表达能力与 softmax 注意力相比如何？
3. 哪些场景下线性注意力更有优势？

---

### 2.2 位置编码（Positional Encoding）

#### 🔥 旋转位置编码 RoPE（rope.md）
**面试要点：**
- 成对旋转编码相对位置的数学原理
- `angles = pos * 1/(10000^(2i/D))` 的频率设计
- 保持向量范数不变的性质
- 点积仅依赖相对位置的证明

**深挖问题：**
1. RoPE 如何编码相对位置？请推导数学公式。
2. 为什么选择 10000 作为基底？不同基底有什么影响？
3. RoPE 的长度外推问题是什么？
4. RoPE 与 ALiBi 的对比优劣？

#### 🔥 NTK-aware RoPE（ntk_rope.md）
**面试要点：**
- 长上下文外推的核心问题
- NTK-aware 缩放调整基底频率：`base_new = 10000 * scale^(D/(D-2))`
- 高频维度保持、低频维度拉伸的特性
- 无需微调即可外推的能力

**深挖问题：**
1. 为什么直接缩放 RoPE 会导致性能下降？
2. NTK-aware 缩放如何区分高频和低频维度？
3. 与 YaRN、Dynamic NTK 等其他外推方法的对比？
4. 实际应用中如何选择缩放比例？

#### ⭐ ALiBi（alibi.md）
**面试要点：**
- 用固定线性偏置替代位置嵌入
- 每个头不同的斜率：`m_h = 1 / 2^(8h/H)`
- 对远距离 token 注意力的惩罚机制
- 训练长度外推能力

**深挖问题：**
1. ALiBi 的斜率设计有什么考虑？
2. 为什么 ALiBi 有较好的长度外推能力？
3. ALiBi 与 RoPE 在不同任务上的表现差异？
4. 为什么现代 LLM 更多选择 RoPE 而不是 ALiBi？

#### 📚 正弦位置编码（sinusoidal_pe.md）
**面试要点：**
- 原始 Transformer 的位置编码方案
- sin/cos 交替编码的设计思想
- 相对位置可以通过线性变换获得

---

### 2.3 训练技术（Training Techniques）

#### 🔥 Adam 优化器（adam.md）
**面试要点：**
- 一阶矩（动量）和二阶矩（自适应学习率）的结合
- 偏差校正的必要性：`m_hat = m / (1 - beta1^t)`
- 与 SGD、RMSProp 的对比
- AdamW 的权重衰减改进

**深挖问题：**
1. 为什么 Adam 需要偏差校正？
2. beta1 和 beta2 的默认值分别是多少？为什么选择这些值？
3. Adam 在什么情况下可能失效？（如稀疏梯度）
4. AdamW 与 Adam 的区别是什么？为什么 AdamW 更常用？

#### 🔥 混合精度训练（mixed_precision.md）
**面试要点：**
- fp16 前向/反向传播 + fp32 主权重
- Loss Scaling 防止梯度下溢
- 动态损失缩放策略
- 与 bf16 的对比

**深挖问题：**
1. 为什么需要 Loss Scaling？fp16 的梯度范围有什么问题？
2. 如何检测和处理梯度溢出（NaN/Inf）？
3. bf16 相比 fp16 有什么优势？为什么现代 GPU 更推荐 bf16？
4. 混合精度训练的内存节省如何计算？

#### ⭐ 梯度累积（gradient_accumulation.md）
**面试要点：**
- 模拟大批次训练的技术
- 每个微批次损失除以 n 的必要性
- 与数据并行的关系
- 实际训练中的 batch size 计算

**深挖问题：**
1. 为什么每个微批次的 loss 要除以 n？
2. 梯度累积与数据并行有什么区别？
3. 实际训练中如何设置 micro_batch_size 和 accumulation_steps？
4. 梯度累积对 Batch Normalization 有什么影响？

#### ⭐ 激活检查点（activation_checkpointing.md）
**面试要点：**
- 以计算换内存的策略
- `torch.utils.checkpoint.checkpoint` 的使用
- 前向传播不存储中间激活，反向传播时重新计算
- 内存节省与计算开销的权衡

**深挖问题：**
1. 激活检查点如何影响训练速度？
2. 应该在哪些层使用激活检查点？
3. 与 ZeRO 等分布式策略的配合使用？
4. 激活检查点的实现原理是什么？（autograd 的 custom function）

#### ⭐ 梯度裁剪（gradient_clipping.md）
**面试要点：**
- 防止梯度爆炸的技术
- L2 范数的计算：`sqrt(sum(p.grad.norm()^2))`
- 只在超过阈值时裁剪，保持梯度方向不变
- 与梯度缩放的区别

**深挖问题：**
1. 为什么选择 L2 范数而不是 L1 范数？
2. max_norm 应该设置为多少？如何调整？
3. 梯度裁剪对训练动态有什么影响？
4. 什么时候应该使用梯度裁剪？

---

### 2.4 对齐技术（Alignment Techniques）

#### 🔥 DPO 损失（dpo_loss.md）
**面试要点：**
- 直接偏好优化：无需强化学习的对齐方法
- 使用配对的选中/拒绝对数概率
- 与 RLHF 的对比优势
- beta 参数的温度缩放作用

**深挖问题：**
1. DPO 的数学推导过程是什么？（从 RLHF 目标到 DPO 损失）
2. DPO 相比 PPO 有什么优势和劣势？
3. beta 参数如何影响训练效果？
4. DPO 的参考模型有什么作用？可以去掉吗？
5. DPO 在什么情况下可能失效？

#### 🔥 GRPO 损失（grpo_loss.md）
**面试要点：**
- 组相对策略优化：DeepSeek 的对齐方法
- 组内 z-score 归一化奖励
- 无需价值网络（Critic）
- 与 PPO 的对比优势

**深挖问题：**
1. GRPO 为什么不需要价值网络？
2. 组内归一化有什么优势？
3. GRPO 与 DPO 的适用场景对比？
4. DeepSeek 为什么选择 GRPO 而不是 DPO？

#### 🔥 PPO 损失（ppo_loss.md）
**面试要点：**
- 裁剪代理损失防止破坏性更新
- 重要性采样比率：`r = exp(new_logps - old_logps.detach())`
- 裁剪范围 epsilon 的作用
- 与 TRPO 的关系

**深挖问题：**
1. PPO 的裁剪是如何工作的？
2. 为什么需要重要性采样？
3. PPO 的训练稳定性如何保证？
4. PPO 在 LLM 对齐中的具体实现细节？

#### 🔥 奖励模型（reward_model.md）
**面试要点：**
- Bradley-Terry 偏好模型
- 从隐藏状态到标量奖励的投影
- 偏好损失：`-log(sigmoid(r_chosen - r_rejected))`
- 手动实现 sigmoid 的约束

**深挖问题：**
1. 奖励模型如何收集训练数据？
2. 奖励模型的过拟合问题如何解决？
3. 奖励模型的奖励分数有界吗？
4. 如何评估奖励模型的质量？

#### ⭐ 交叉熵损失（cross_entropy.md）
**面试要点：**
- 分类任务的标准损失
- log-sum-exp 技巧保证数值稳定性
- 与 softmax 的关系
- 标签平滑的改进

#### ⭐ 标签平滑（label_smoothing.md）
**面试要点：**
- 防止过度自信的正则化技术
- 软目标分布：正确类别 `1-ε`，其他类别 `ε/(C-1)`
- Transformer 训练中的广泛应用
- 对模型校准的影响

#### ⭐ Focal Loss（focal_loss.md）
**面试要点：**
- 处理类别不平衡的损失函数
- 降低简单样本权重：`(1-p_t)^gamma`
- RetinaNet 中的提出背景
- gamma 参数的调节作用

#### 📚 对比损失（contrastive_loss.md）
**面试要点：**
- InfoNCE / NT-Xent 损失
- CLIP 和 SimCLR 中的应用
- 正样本拉近、负样本推远
- 温度参数的作用

---

### 2.5 推理优化（Inference Optimization）

#### 🔥 KV Cache（kv_cache.md）
**面试要点：**
- 自回归解码的核心优化
- 缓存之前计算的键和值，只处理新 token
- 预填充（Prefill）和解码（Decode）阶段的区别
- 因果掩码的正确实现

**深挖问题：**
1. KV Cache 的内存占用如何计算？
2. 如何处理变长序列的 KV Cache？
3. KV Cache 的生命周期管理？
4. 与 Paged Attention 的配合使用？

#### 🔥 推测解码（speculative_decoding.md）
**面试要点：**
- 快速草稿模型提议 + 目标模型验证
- 接受概率：`min(1, p_target/p_draft)`
- 拒绝时的重采样：`max(0, p_target - p_draft)`
- 保持分布一致性的保证

**深挖问题：**
1. 推测解码为什么能加速推理？
2. 草稿模型如何选择？
3. 接受率如何影响加速比？
4. 与 Medusa、EAGLE 等方法的对比？

#### 🔥 Top-k / Top-p 采样（topk_sampling.md）
**面试要点：**
- 温度缩放、Top-k 过滤、Top-p 截断
- 采样顺序：温度 → Top-k → Top-p
- 多样性与质量的平衡
- `top_k=1` 时退化为贪心解码

**深挖问题：**
1. 温度参数如何影响生成质量？
2. Top-k 和 Top-p 各自的优缺点？
3. 实际应用中如何设置这些参数？
4. 与 Repetition Penalty 等其他解码策略的配合？

#### 🔥 分页注意力（paged_attention.md）
**面试要点：**
- vLLM 的核心创新
- KV Cache 的非连续内存管理
- Block Table 的逻辑到物理映射
- 减少内存碎片化

**深挖问题：**
1. 分页注意力如何减少内存碎片？
2. Block Table 的设计细节？
3. 与连续 KV Cache 的性能对比？
4. vLLM 的其他优化技术？（Continuous Batching 等）

#### ⭐ 束搜索（beam_search.md）
**面试要点：**
- 维护多个候选序列的搜索策略
- 扩展和剪枝的平衡
- 与贪心解码、采样的对比
- 长度惩罚的引入

#### ⭐ BPE 分词（bpe.md）
**面试要点：**
- 字节对编码的训练过程
- 子词词表的构建
- `</w>` 词边界标记
- 与 WordPiece、SentencePiece 的对比

---

### 2.6 参数高效微调（PEFT）

#### 🔥 LoRA（lora.md）
**面试要点：**
- 低秩适配：冻结基础权重 + 可训练的 A、B 矩阵
- 缩放因子 `alpha/rank` 的作用
- 零初始化 B 矩阵的必要性
- 可训练参数量的计算

**深挖问题：**
1. LoRA 的数学原理是什么？为什么低秩假设有效？
2. rank 和 alpha 如何选择？
3. LoRA 应该加在哪些层？为什么？
4. LoRA 与 Full Fine-tuning 的效果对比？
5. LoRA 的合并（merge）操作是什么？

#### 🔥 QLoRA（qlora.md）
**面试要点：**
- 量化基础权重 + LoRA 适配器
- int8 量化 + 每行缩放因子
- 反量化计算：`W_fp = quantized_weight.float() * scale`
- 4-bit 量化的实际实现

**深挖问题：**
1. QLoRA 的内存节省如何计算？
2. 量化对模型质量有什么影响？
3. NF4 数据类型是什么？
4. QLoRA 的训练速度与 LoRA 相比如何？

---

### 2.7 分布式训练（Distributed Training）

#### 🔥 张量并行（tensor_parallel.md）
**面试要点：**
- Megatron 风格的 MLP 并行
- 列并行（Column Parallel）和行并行（Row Parallel）
- All-reduce 操作的实现
- 与模型并行、数据并行的区别

**深挖问题：**
1. 列并行和行并行分别适用于哪些层？
2. 通信量如何计算？
3. 张量并行与流水线并行的对比？
4. 实际训练中如何设置并行策略？

#### ⭐ FSDP（fsdp_step.md）
**面试要点：**
- 全分片数据并行的核心思想
- All-gather 和 Reduce-scatter 操作
- 与 ZeRO 的关系
- 内存节省的计算

**深挖问题：**
1. FSDP 的三个阶段分别是什么？
2. FSDP 与 DDP 的对比？
3. 如何处理不同大小的参数？
4. FSDP 的通信开销如何优化？

#### ⭐ 环形注意力（ring_attention.md）
**面试要点：**
- 序列并行的实现
- Q/K/V 在环形拓扑上的轮转
- 在线 softmax 的累加
- 超长序列训练的解决方案

---

### 2.8 架构组件（Architectural Components）

#### 🔥 RMSNorm（rmsnorm.md）
**面试要点：**
- 简化的 LayerNorm，跳过均值减法
- 计算效率提升：`x / RMS(x) * weight`
- LLaMA 等现代 LLM 的选择
- 与 LayerNorm 的对比

**深挖问题：**
1. 为什么 RMSNorm 比 LayerNorm 更高效？
2. RMSNorm 对训练稳定性有什么影响？
3. 为什么现代 LLM 普遍选择 RMSNorm？
4. RMSNorm 的梯度计算有什么特点？

#### 🔥 SwiGLU 激活函数（swiglu.md, mlp.md）
**面试要点：**
- 门控激活函数：`(x @ W1) * swish(x @ Wgate)`
- LLaMA、PaLM 等模型的选择
- 与 ReLU、GELU 的对比
- 参数量增加 50% 的原因

**深挖问题：**
1. SwiGLU 为什么比 ReLU 效果好？
2. SwiGLU 的参数量如何计算？
3. 为什么 d_ff 通常设置为 2/3 * 4d？
4. GLU 变体有哪些？为什么 SwiGLU 最常用？

#### ⭐ GPT-2 Block（gpt2_block.md）
**面试要点：**
- Pre-norm 架构：`x = x + attn(ln1(x))`
- 因果掩码的实现
- MLP：`Linear(d, 4d) -> GELU -> Linear(4d, d)`
- 残差连接的重要性

#### ⭐ MoE（moe.md, moe_load_balance.md）
**面试要点：**
- 混合专家的路由机制
- Top-k 专家选择
- 负载均衡损失的必要性
- 与稠密模型的对比

**深挖问题：**
1. MoE 的路由是如何学习的？
2. 为什么需要负载均衡损失？
3. MoE 的训练稳定性问题？
4. Switch Transformer 和 DeepSeek MoE 的区别？

#### 📚 激活函数（relu.md, gelu.md）
**面试要点：**
- ReLU：`max(0, x)` 的简洁实现
- GELU：`x * 0.5 * (1 + erf(x / sqrt(2)))`
- 梯度流的特性
- 在 Transformer 中的应用

---

### 2.9 扩散模型（Diffusion Models）

#### ⭐ 噪声调度（noise_schedule.md）
**面试要点：**
- 线性、余弦、sigmoid 三种调度
- alpha_bar 控制信噪比
- 扩散过程的数学描述
- Stable Diffusion 中的应用

#### ⭐ DDIM 采样（ddim_step.md）
**面试要点：**
- 确定性采样的实现
- 预测 x0 的公式
- 与 DDPM 的对比
- 加速采样的原理

#### ⭐ 流匹配（flow_matching.md）
**面试要点：**
- Stable Diffusion 3、Flux、Sora 的核心范式
- 直线路径从噪声到数据
- 速度场预测
- 与扩散模型的对比

---

### 2.10 Agent 与推理（Agent & Reasoning）

#### 🔥 MCTS 搜索（mcts_search.md）
**面试要点：**
- o1/AlphaProof 风格推理的核心算法
- UCB1 选择策略：`Q(i) + c_puct * sqrt(log(parent_visits) / visit_counts[i])`
- 模拟和反向传播
- 探索与利用的平衡

**深挖问题：**
1. MCTS 如何与 LLM 结合？
2. UCB1 中 c_puct 如何调节？
3. 与 Chain-of-Thought 的对比？
4. AlphaProof 是如何使用 MCTS 的？

#### ⭐ 多 Token 预测（multi_token_prediction.md）
**面试要点：**
- Meta 2024 年论文的核心思想
- N 个独立预测头同时预测多个 token
- 手动实现交叉熵的约束
- 对代码生成等任务的提升

**深挖问题：**
1. 多 Token 预测为什么能提升代码生成？
2. 如何处理不同预测头之间的依赖？
3. 推理时如何使用多 Token 预测？
4. 与 Speculative Decoding 的关系？

---

### 2.11 新兴架构（Emerging Architectures）

#### ⭐ Mamba SSM（mamba_ssm.md）
**面试要点：**
- 选择性状态空间模型的核心创新
- B、C、delta 输入相关（选择性）
- 零阶保持离散化
- 递推公式：`h_new = dA * h + dB * x`

**深挖问题：**
1. Mamba 与 Transformer 的对比优劣？
2. 选择性机制为什么重要？
3. Mamba 的长序列处理能力如何？
4. Jamba 等混合架构的设计思路？

---

### 2.12 图神经网络（GNN）

#### 📚 GCN、GAT、GIN、MPNN、GraphSAGE
**面试要点：**
- 图卷积的不同变体
- 邻居聚合的多种方式
- 消息传递框架
- 图级读出操作

**深挖问题：**
1. GCN 的对称归一化为什么重要？
2. GAT 的注意力机制与 Transformer 的区别？
3. 为什么 GIN 的表达能力最强？
4. GraphSAGE 的采样策略有什么优势？

---

## 3. 深度挖掘问题与考察点

### 3.1 注意力机制深度问题

**问题 1：Flash Attention 的在线 softmax 是如何工作的？**

考察点：
- softmax 的数值稳定性处理
- 分块计算的数学正确性
- 内存与计算的权衡

参考答案要点：
1. 维护运行最大值 m 和归一化因子 l
2. 每个新块更新：`m_new = max(m, block_max)`
3. 修正之前累加器：`acc = acc * exp(m - m_new)`
4. 更新归一化因子：`l = l * exp(m - m_new) + exp(scores - m_new).sum()`

**问题 2：GQA 的 KV Cache 大小如何计算？**

考察点：
- KV Cache 的内存占用分析
- 与 MHA、MQA 的对比
- 实际部署中的优化

参考答案要点：
- MHA: `2 * num_layers * seq_len * num_heads * d_head * dtype_size`
- GQA: `2 * num_layers * seq_len * num_kv_heads * d_head * dtype_size`
- MQA: `2 * num_layers * seq_len * 1 * d_head * dtype_size`

**问题 3：MLA 为什么能大幅减少 KV Cache？**

考察点：
- 低秩压缩的数学原理
- 推理时的计算流程
- 与 GQA 的对比分析

参考答案要点：
1. KV Cache 只需存储低秩潜在向量 `c_kv`
2. `c_kv` 的维度远小于完整的 K、V
3. 推理时即时解压 K、V
4. 内存节省：`kv_rank / (num_heads * d_head)`

### 3.2 对齐技术深度问题

**问题 4：DPO 的数学推导过程是什么？**

考察点：
- 从 RLHF 目标到 DPO 损失的推导
- 最优策略的闭式解
- 参考模型的作用

参考答案要点：
1. RLHF 目标：`max E[reward(x,y)] - beta * KL(pi || pi_ref)`
2. 最优策略：`pi*(y|x) = pi_ref(y|x) * exp(reward(x,y)/beta) / Z(x)`
3. 反解 reward：`reward(x,y) = beta * log(pi(y|x)/pi_ref(y|x)) + beta * log(Z(x))`
4. 代入 Bradley-Terry 模型，Z(x) 消除
5. 得到 DPO 损失

**问题 5：GRPO 与 PPO 的核心区别是什么？**

考察点：
- 两种算法的实现细节
- 价值网络的作用
- 训练稳定性对比

参考答案要点：
1. GRPO 无需价值网络（Critic）
2. GRPO 使用组内归一化代替 GAE
3. GRPO 的训练更稳定
4. GRPO 的计算成本更低

**问题 6：奖励模型的训练数据如何收集？**

考察点：
- RLHF 的数据收集流程
- 人类偏好标注的方法
- 数据质量控制

参考答案要点：
1. 采样多个回复，人工排序
2. 配对比较法：选中 vs 拒绝
3. 数据质量：标注一致性、多样性
4. 迭代式数据收集

### 3.3 推理优化深度问题

**问题 7：推测解码为什么能保持分布不变？**

考察点：
- 接受-拒绝采样的数学证明
- 概率比率的作用
- 重采样的正确性

参考答案要点：
1. 接受概率：`min(1, p_target/p_draft)`
2. 拒绝时从 `max(0, p_target - p_draft)` 重采样
3. 数学证明：接受和拒绝的组合恢复目标分布
4. 期望采样长度与 KL 散度的关系

**问题 8：Paged Attention 如何减少内存碎片？**

考察点：
- 传统 KV Cache 的内存管理问题
- 分页的思想借鉴
- Block Table 的设计

参考答案要点：
1. 传统方式：预分配最大长度，浪费内存
2. Paged Attention：固定大小的页，按需分配
3. Block Table：逻辑到物理的映射
4. 减少碎片化，提高内存利用率

### 3.4 训练技术深度问题

**问题 9：混合精度训练中 Loss Scaling 的原理是什么？**

考察点：
- fp16 的数值范围
- 梯度下溢问题
- 动态损失缩放策略

参考答案要点：
1. fp16 最小正规数：`2^-14 ≈ 6e-5`
2. 小梯度可能下溢为 0
3. Loss Scaling 放大梯度，更新时缩小
4. 动态调整缩放因子，检测溢出

**问题 10：激活检查点的实现原理是什么？**

考察点：
- PyTorch autograd 的机制
- Custom Function 的使用
- 内存与计算的权衡

参考答案要点：
1. 前向传播时不保存中间激活
2. 反向传播时重新计算激活
3. 使用 `torch.utils.checkpoint.checkpoint`
4. 内存节省：O(L) → O(√L)

### 3.5 参数高效微调深度问题

**问题 11：LoRA 的 rank 应该如何选择？**

考察点：
- rank 与模型容量的关系
- 不同任务的需求差异
- 实际调参经验

参考答案要点：
1. 通常 rank=8 或 16
2. 简单任务可以用更小的 rank
3. 复杂任务可能需要更大的 rank
4. 需要实验验证

**问题 12：QLoRA 的 NF4 数据类型是什么？**

考察点：
- 量化的数据类型
- NF4 的设计思想
- 与 INT4、FP4 的对比

参考答案要点：
1. NF4：NormalFloat 4-bit
2. 基于正态分布的量化
3. 信息论最优的 4-bit 数据类型
4. 与 INT4 的精度对比

---

## 4. 综合应用场景与系统设计

### 4.1 设计一个 LLM 推理系统

**考察点：**
- KV Cache 管理
- 批处理策略
- 内存优化

**设计要点：**
1. **KV Cache 管理**：使用 Paged Attention，Block Table 映射
2. **批处理**：Continuous Batching，动态插入新请求
3. **内存优化**：GQA 减少 KV Cache，INT8 量化
4. **加速**：推测解码、Flash Attention

**深挖问题：**
1. 如何处理不同长度的请求？
2. 如何实现抢占式调度？
3. 如何优化长序列的推理？
4. 如何实现流式输出？

### 4.2 设计一个 LLM 训练系统

**考察点：**
- 分布式训练策略
- 内存优化技术
- 训练稳定性

**设计要点：**
1. **分布式**：FSDP + 张量并行
2. **内存优化**：激活检查点、混合精度
3. **训练稳定性**：梯度裁剪、学习率预热
4. **效率**：梯度累积、数据并行

**深挖问题：**
1. 如何选择并行策略？
2. 如何处理训练中的 NaN？
3. 如何监控训练状态？
4. 如何实现断点续训？

### 4.3 设计一个对齐训练流程

**考察点：**
- RLHF/DPO 的完整流程
- 数据收集与处理
- 模型评估

**设计要点：**
1. **SFT**：监督微调，基础对话能力
2. **奖励模型**：Bradley-Terry 模型训练
3. **对齐训练**：DPO/GRPO/PPO
4. **评估**：自动评估 + 人工评估

**深挖问题：**
1. 如何收集高质量的偏好数据？
2. 如何选择对齐算法？
3. 如何防止对齐税（Alignment Tax）？
4. 如何评估对齐效果？

### 4.4 设计一个 Agent 系统

**考察点：**
- LLM 与工具的结合
- 推理与规划
- 多轮交互

**设计要点：**
1. **推理**：Chain-of-Thought、MCTS
2. **工具使用**：Function Calling、Toolformer
3. **规划**：任务分解、子目标设定
4. **记忆**：长期记忆、短期记忆

**深挖问题：**
1. 如何设计工具接口？
2. 如何处理工具调用失败？
3. 如何实现多步推理？
4. 如何评估 Agent 性能？

---

## 5. 面试技巧与准备建议

### 5.1 面试准备策略

**第一阶段：基础巩固（1-2 周）**
1. 完成所有 Easy 题目，确保基础扎实
2. 理解每个组件的数学原理
3. 能够手写核心代码

**第二阶段：深度理解（2-3 周）**
1. 完成 Medium 和 Hard 题目
2. 理解设计决策和权衡
3. 能够回答深挖问题

**第三阶段：系统整合（1 周）**
1. 理解组件之间的关系
2. 能够设计完整系统
3. 准备项目经验的讲述

### 5.2 面试回答技巧

**STAR 法则：**
- **Situation**：描述背景和问题
- **Task**：说明你的任务和目标
- **Action**：详细说明你的行动
- **Result**：总结结果和收获

**代码讲解模板：**
1. 先说整体思路
2. 再讲关键实现
3. 最后说优化和权衡

**深挖问题应对：**
1. 不知道就说不知道，但尝试推理
2. 从原理出发，逐步推导
3. 举例说明，具体化抽象概念

### 5.3 常见面试问题清单

**基础问题：**
1. 请介绍 Transformer 的注意力机制
2. RoPE 是如何编码位置信息的？
3. 什么是 KV Cache？为什么需要它？
4. 解释 LoRA 的原理和优势

**进阶问题：**
1. Flash Attention 的在线 softmax 是如何工作的？
2. DPO 的数学推导过程是什么？
3. 推测解码为什么能保持分布不变？
4. MLA 为什么能减少 KV Cache？

**系统设计问题：**
1. 如何设计一个 LLM 推理系统？
2. 如何实现分布式训练？
3. 如何设计一个对齐训练流程？
4. 如何优化长序列的推理？

### 5.4 项目经验讲述

**推荐讲述结构：**
1. **背景**：为什么要做这个项目？
2. **挑战**：遇到了什么技术挑战？
3. **方案**：你提出了什么解决方案？
4. **实现**：具体如何实现的？
5. **结果**：取得了什么效果？
6. **反思**：有什么经验教训？

**技术深度体现：**
1. 能够解释设计决策的原因
2. 能够讨论权衡和替代方案
3. 能够分析性能瓶颈和优化
4. 能够总结可复用的经验

### 5.5 持续学习建议

**论文阅读：**
1. Flash Attention 1/2
2. LLaMA 1/2/3
3. DPO、GRPO 论文
4. vLLM、Speculative Decoding

**开源项目：**
1. Hugging Face Transformers
2. vLLM
3. DeepSpeed
4. Megatron-LM

**技术博客：**
1. Lilian Weng 的博客
2. Jay Alammar 的可视化
3. Sebastian Raschka 的博客

---

## 📊 题目难度分布与面试频率

| 类别 | Easy | Medium | Hard | 面试频率 |
|------|------|--------|------|----------|
| 注意力机制 | 2 | 3 | 6 | 🔥🔥🔥 |
| 位置编码 | 1 | 2 | 1 | 🔥🔥🔥 |
| 训练技术 | 3 | 4 | 1 | 🔥🔥 |
| 对齐技术 | 3 | 3 | 1 | 🔥🔥🔥 |
| 推理优化 | 1 | 2 | 4 | 🔥🔥🔥 |
| 参数高效微调 | 0 | 1 | 1 | 🔥🔥 |
| 分布式训练 | 0 | 1 | 2 | 🔥🔥 |
| 架构组件 | 3 | 3 | 2 | 🔥🔥 |
| 扩散模型 | 1 | 2 | 0 | 🔥 |
| Agent 推理 | 0 | 0 | 2 | 🔥🔥 |
| 图神经网络 | 2 | 3 | 1 | 🔥 |

---

## 🎯 面试重点总结

### 必须掌握的核心题目（🔥）
1. Flash Attention - 在线 softmax、分块处理
2. RoPE - 旋转位置编码、长度外推
3. GQA/MLA - KV Cache 优化
4. DPO/GRPO - 对齐算法
5. KV Cache - 推理基础
6. 推测解码 - 推理加速
7. LoRA/QLoRA - 参数高效微调
8. Adam - 优化器
9. 混合精度训练 - 训练优化
10. MCTS - Agent 推理

### 需要深入理解的原理
1. Softmax 的数值稳定性处理
2. 注意力机制的复杂度分析
3. 位置编码的数学原理
4. 对齐算法的推导过程
5. 分布式训练的通信分析

### 需要掌握的系统设计
1. LLM 推理系统设计
2. 分布式训练系统设计
3. 对齐训练流程设计
4. Agent 系统设计

---

## 📚 推荐阅读材料

### 论文
1. **Attention Is All You Need** - Transformer 原始论文
2. **Flash Attention** - Flash Attention 1/2
3. **RoPE** - 旋转位置编码
4. **DPO** - 直接偏好优化
5. **GRPO** - 组相对策略优化
6. **vLLM** - Paged Attention
7. **Speculative Decoding** - 推测解码
8. **LoRA** - 低秩适配
9. **Mamba** - 选择性状态空间模型
10. **DeepSeek V2/V3** - MLA 和 MoE

### 书籍
1. **Deep Learning** - Ian Goodfellow
2. **Speech and Language Processing** - Dan Jurafsky
3. **Reinforcement Learning: An Introduction** - Richard Sutton

### 开源项目
1. **Hugging Face Transformers** - 模型实现
2. **vLLM** - 推理优化
3. **DeepSpeed** - 分布式训练
4. **Megatron-LM** - 大模型训练
5. **trl** - 对齐训练

---

**最后更新：2026 年 5 月 27 日**
**基于 Pyre Code 项目 76 道 ML 编程题目整理**