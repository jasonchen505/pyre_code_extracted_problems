# Paged Attention

**中文标题:** 分页注意力

**难度:** Hard

**函数名:** `paged_attention`

## 英文描述

Implement paged attention (vLLM-style).

Instead of a contiguous KV cache, keys and values are stored in fixed-size pages (blocks). A block table maps logical block indices to physical page indices, enabling non-contiguous memory allocation.

**Signature:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`

**Parameters:**
- `Q` — query tensor `(B, 1, D)` (single decode step)
- `k_pages` — key pages `(num_pages, block_size, D)`
- `v_pages` — value pages `(num_pages, block_size, D)`
- `block_table` — integer tensor `(B, max_blocks)` mapping logical → physical page
- `context_len` — number of valid KV tokens per sequence (scalar)
- `block_size` — tokens per page

**Returns:** output tensor `(B, 1, D)`

**Steps:** Use block_table to gather K/V pages, slice to context_len, then compute standard scaled dot-product attention.

## 中文描述

实现分页注意力（vLLM 风格）。

KV 缓存不再连续存储，而是存在固定大小的页（block）中。block table 将逻辑块索引映射到物理页索引，实现非连续内存分配。

**签名:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`

**参数:**
- `Q` — 查询张量 `(B, 1, D)`（单步解码）
- `k_pages` — 键页 `(num_pages, block_size, D)`
- `v_pages` — 值页 `(num_pages, block_size, D)`
- `block_table` — 整数张量 `(B, max_blocks)`，逻辑块 → 物理页
- `context_len` — 每条序列有效 KV token 数（标量）
- `block_size` — 每页 token 数

**返回:** 输出张量 `(B, 1, D)`

**步骤:** 用 block_table gather K/V 页，截取到 context_len，再做标准缩放点积注意力。

## 提示

```
for b in range(B):
  K = k_pages[block_table[b]].reshape(-1, D)[:context_len]  # gather + slice
  V = v_pages[block_table[b]].reshape(-1, D)[:context_len]
  scores = Q[b] @ K.T / √D → softmax → @ V
```

## 参考答案

```python
def paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size):
    B, _, D = Q.shape
    outputs = []
    for b in range(B):
        # Gather K/V pages for this sequence
        phys_blocks = block_table[b]  # (max_blocks,)
        K_gathered = k_pages[phys_blocks].reshape(-1, D)  # (max_blocks*block_size, D)
        V_gathered = v_pages[phys_blocks].reshape(-1, D)
        # Slice to actual context length
        K_ctx = K_gathered[:context_len].unsqueeze(0)  # (1, context_len, D)
        V_ctx = V_gathered[:context_len].unsqueeze(0)
        # Scaled dot-product attention
        scale = D ** -0.5
        scores = (Q[b:b+1] @ K_ctx.transpose(-2, -1)) * scale  # (1, 1, context_len)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V_ctx  # (1, 1, D)
        outputs.append(out)
    return torch.cat(outputs, dim=0)
```

## 示例代码

```python
torch.manual_seed(0)
B, S, D = 2, 8, 16
block_size = 4
num_blocks = S // block_size

K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)

scale = D ** -0.5
scores_ref = (Q @ K_full.transpose(-2, -1)) * scale
ref_out = torch.softmax(scores_ref, dim=-1) @ V_full

total_pages = B * num_blocks
k_pages = torch.zeros(total_pages, block_size, D)
v_pages = torch.zeros(total_pages, block_size, D)
block_table = []
for b in range(B):
    page_ids = []
    for blk in range(num_blocks):
        pid = b * num_blocks + blk
        k_pages[pid] = K_full[b, blk*block_size:(blk+1)*block_size]
        v_pages[pid] = V_full[b, blk*block_size:(blk+1)*block_size]
        page_ids.append(pid)
    block_table.append(page_ids)

paged_out = paged_attention(Q, k_pages, v_pages, block_table, context_len=S, block_size=block_size)

print('Shape:', paged_out.shape)
print('Max diff vs reference:', (paged_out - ref_out).abs().max().item())
print('Match:', torch.allclose(paged_out, ref_out, atol=1e-5))
```

