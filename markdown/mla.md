# Multi-Head Latent Attention (MLA)

**中文标题:** 多头潜在注意力（MLA）

**难度:** Hard

**函数名:** `mla_attention`

## 英文描述

Implement Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.

MLA's key innovation: instead of caching full K and V tensors, compress them into a low-rank latent vector `c_kv`, then decompress on the fly. This dramatically reduces KV cache memory during inference.

**Signature:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`

**Parameters:**
- `X` — input tensor (B, S, D)
- `W_dkv` — KV compression matrix (D, kv_rank)
- `W_uk` — K decompression matrix (kv_rank, num_heads * D_h)
- `W_uv` — V decompression matrix (kv_rank, num_heads * D_h)
- `W_q` — Q projection matrix (D, num_heads * D_h)
- `num_heads` — number of attention heads

**Returns:** output tensor (B, S, num_heads * D_h)

**Steps:**
1. Compress: `c_kv = X @ W_dkv` → (B, S, kv_rank)
2. Decompress: `K = c_kv @ W_uk`, `V = c_kv @ W_uv`
3. Project queries: `Q = X @ W_q`
4. Reshape all to (B, num_heads, S, D_h)
5. Scaled dot-product attention
6. Reshape output to (B, S, num_heads * D_h)

## 中文描述

实现 DeepSeek V2/V3 中的多头潜在注意力（MLA）。

MLA 的核心创新：不缓存完整的 K 和 V 张量，而是将其压缩为低秩潜在向量 `c_kv`，推理时再即时解压。这大幅降低了推理时的 KV 缓存内存占用。

**签名:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`

**参数:**
- `X` — 输入张量 (B, S, D)
- `W_dkv` — KV 压缩矩阵 (D, kv_rank)
- `W_uk` — K 解压矩阵 (kv_rank, num_heads * D_h)
- `W_uv` — V 解压矩阵 (kv_rank, num_heads * D_h)
- `W_q` — Q 投影矩阵 (D, num_heads * D_h)
- `num_heads` — 注意力头数

**返回:** 输出张量 (B, S, num_heads * D_h)

**步骤:**
1. 压缩：`c_kv = X @ W_dkv` → (B, S, kv_rank)
2. 解压：`K = c_kv @ W_uk`，`V = c_kv @ W_uv`
3. 投影查询：`Q = X @ W_q`
4. 重塑为 (B, num_heads, S, D_h)
5. 缩放点积注意力
6. 重塑输出为 (B, S, num_heads * D_h)

## 提示

```
1. `c_kv = X @ W_dkv`  → `(B,S,kv_rank)`
2. `K = c_kv @ W_uk`, `V = c_kv @ W_uv`, `Q = X @ W_q`
3. `.view(B,S,num_heads,D_h).transpose(1,2)` for each
4. Scaled dot-product attn → `.transpose(1,2).reshape(B,S,-1)`
```

## 参考答案

```python
def mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads):
    B, S, D = X.shape
    D_h = W_q.shape[1] // num_heads
    # Compress KV into low-rank latent
    c_kv = X @ W_dkv                          # (B, S, kv_rank)
    K = c_kv @ W_uk                            # (B, S, num_heads*D_h)
    V = c_kv @ W_uv                            # (B, S, num_heads*D_h)
    Q = X @ W_q                                # (B, S, num_heads*D_h)
    # Reshape to multi-head format
    def split_heads(t):
        return t.view(B, S, num_heads, D_h).transpose(1, 2)
    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
    scale = D_h ** -0.5
    attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    out = (attn @ V).transpose(1, 2).reshape(B, S, num_heads * D_h)
    return out
```

## 示例代码

```python
torch.manual_seed(0)
B, S, D = 2, 6, 32
num_heads = 4
D_h = 8          # head dim
D_c = 8          # compressed KV dim (latent)

W_dkv = torch.randn(D, D_c) * 0.1      # compress to latent
W_uk  = torch.randn(D_c, num_heads * D_h) * 0.1  # up-project to K
W_uv  = torch.randn(D_c, num_heads * D_h) * 0.1  # up-project to V
W_q   = torch.randn(D, num_heads * D_h) * 0.1

X = torch.randn(B, S, D)

c_kv = X @ W_dkv
K_full = c_kv @ W_uk
print(f"Input shape:          {X.shape}")       # (2, 6, 32)
print(f"Compressed KV shape:  {c_kv.shape}")    # (2, 6, 8)  <-- small latent
print(f"Full K shape:         {K_full.shape}")  # (2, 6, 32) <-- expanded

out = mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads)
print(f"Output shape:         {out.shape}")     # (2, 6, 32)
```

