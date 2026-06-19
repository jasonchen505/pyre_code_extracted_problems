# 提取的面试题目

本目录包含从 [pyre-code](https://github.com/whwangovo/pyre-code) 项目中提取的 76 道 ML 面试题目及参考答案。

## 目录结构

```
extracted_problems/
├── problems.json          # 所有题目的 JSON 格式数据
├── markdown/              # 每个题目的独立 Markdown 文件
│   ├── relu.md
│   ├── softmax.md
│   └── ... (共 76 个文件)
└── README_EXTRACTED.md    # 本说明文件
```

## 数据格式

### JSON 文件

每个题目包含以下字段：

- `module`: 模块名（对应源文件名）
- `title`: 英文标题
- `title_zh`: 中文标题
- `difficulty`: 难度（Easy/Medium/Hard）
- `description_en`: 英文描述
- `description_zh`: 中文描述
- `function_name`: 需要实现的函数名
- `hint`: 英文提示
- `hint_zh`: 中文提示
- `solution`: 参考答案（Python 代码）
- `demo`: 示例代码
- `tests`: 测试用例列表

### Markdown 文件

每个 Markdown 文件包含题目的完整信息，格式化为易读的文本。

## 题目分类

共 76 道题目，涵盖以下主题：

- **基础组件**: ReLU, Softmax, GELU, SwiGLU, Dropout, Embedding, Linear, Kaiming Init, Linear Regression
- **归一化**: LayerNorm, BatchNorm, RMSNorm
- **注意力机制**: Scaled Dot-Product, Multi-Head, Causal, Cross, GQA, Sliding Window, Linear, Flash, Differential, MLA
- **位置编码**: Sinusoidal PE, RoPE, ALiBi, NTK-aware RoPE
- **架构**: SwiGLU MLP, GPT-2 Block, ViT Patch, ViT Block, Conv2D, Max Pool, Depthwise Conv, MoE, MoE Load Balance
- **训练**: Adam, Cosine LR, Gradient Clipping, Gradient Accumulation, Mixed Precision, Activation Checkpointing
- **分布式**: Tensor Parallel, FSDP, Ring Attention
- **推理**: KV Cache, Top-k Sampling, Beam Search, Speculative Decoding, BPE, INT8 Quantization, Paged Attention
- **损失函数与对齐**: Cross Entropy, Label Smoothing, Focal Loss, Contrastive Loss, DPO, GRPO, PPO, Reward Model
- **扩散模型**: Noise Schedule, DDIM Step, Flow Matching, adaLN-Zero
- **适配**: LoRA, QLoRA
- **推理**: MCTS, Multi-Token Prediction
- **SSM**: Mamba SSM
- **图神经网络**: GCN, Graph Readout, GAT, GIN, MPNN, GraphSAGE, Link Prediction, Graph Autoencoder

## 使用方法

1. **查看单个题目**: 打开 `markdown/` 目录下的 Markdown 文件
2. **编程练习**: 根据题目描述实现函数，参考 `solution` 字段的答案
3. **批量处理**: 使用 `problems.json` 文件进行数据分析或导入其他系统

## 提取时间

提取于 2026 年 5 月 27 日